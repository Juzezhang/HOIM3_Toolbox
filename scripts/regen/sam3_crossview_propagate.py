"""Cross-view mask propagation using SAM3 video predictor.

Treat 42 cameras as a "video" by sampling 1 frame each at a chosen frame_id,
sorted by camera angular position (so neighboring "frames" are spatially
adjacent views). Initialize SAM3 with text prompt at the user-picked ref view;
SAM3 propagates the detected object identity to all other views.

Usage:
    python sam3_crossview_propagate.py \
        --seq office_data32 --obj shredder \
        --ref_view 35 --ref_frame 10000 \
        --out_dir /simurgh2/datasets/HOI-M3/sam3_xv_probe
"""
import argparse
import json
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np


N_VIEWS = 42


def load_calib_for_seq(seq):
    """Load 42-view camera centers (world coords) to sort views by position."""
    from os.path import join
    DATASET_INFO = "/simurgh/group/juze/datasets/HOI-M3/dataset_information.json"
    ROOT_CALIB = "/simurgh2/datasets/HOI-M3/calib_ground_refined"
    info = json.load(open(DATASET_INFO))
    seq2date = {s: d for d, seqs in info.items() for s in seqs}
    date = seq2date[seq]
    cf = join(ROOT_CALIB, date, "calibration.json")
    raw = json.load(open(cf))
    centers = {}
    for k, v in raw.items():
        try:
            RT = np.asarray(v["RT"], dtype=np.float64).reshape(-1, 4)
            R = RT[:3, :3]; t = RT[:3, 3]
            C = -R.T @ t  # world-coords camera center
            centers[int(k)] = C
        except Exception:
            pass
    return centers


def sort_views_by_angle(centers, ref_view):
    """Sort view ids by angular position around scene centroid, starting from ref_view."""
    cs = np.array([centers[v] for v in sorted(centers.keys()) if centers.get(v) is not None])
    scene_center = cs.mean(axis=0)
    # angle in horizontal plane (assume Y or Z is up; use atan2 of x vs z)
    angles = {}
    for v in centers:
        C = centers[v]
        d = C - scene_center
        angles[v] = float(np.arctan2(d[2], d[0]))
    sorted_ids = sorted(angles.keys(), key=lambda v: angles[v])
    if ref_view in sorted_ids:
        i = sorted_ids.index(ref_view)
        sorted_ids = sorted_ids[i:] + sorted_ids[:i]
    return sorted_ids


def build_video(seq, frame_id, sorted_views, img_root, target_size=(1280, 720)):
    """Materialize a 'video' dir of N_VIEWS jpgs in sorted-view order. Return tmp_dir + map."""
    tmp = tempfile.mkdtemp(prefix=f"sam3_xv_{seq}_")
    frame_to_view = {}
    tw, th = target_size
    written = 0
    for i, v in enumerate(sorted_views):
        ip = os.path.join(img_root, seq, str(v), f"{frame_id:06d}.jpg")
        if not os.path.isfile(ip):
            # If missing, use a black frame so we still preserve the index
            blank = np.zeros((th, tw, 3), dtype=np.uint8)
            out = os.path.join(tmp, f"{i:05d}.jpg")
            cv2.imwrite(out, blank)
        else:
            img = cv2.imread(ip)
            if img.shape[:2] != (th, tw):
                img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)
            out = os.path.join(tmp, f"{i:05d}.jpg")
            cv2.imwrite(out, img)
        frame_to_view[i] = v
        written += 1
    print(f"[xv] wrote {written} frames to {tmp}")
    return tmp, frame_to_view


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--obj", required=True)
    ap.add_argument("--ref_view", type=int, required=True)
    ap.add_argument("--ref_frame", type=int, required=True)
    ap.add_argument("--detection_idx", type=int, default=0,
                    help="Which detection in ref_view to use (0=highest score)")
    ap.add_argument("--fallback_score_thresh", type=float, default=0.40,
                    help="Min SAM3 score for per-view fallback detection")
    ap.add_argument("--fallback_frame_offsets", type=str,
                    default="500,1000,2000,3000,5000,8000,12000,16000,-500,-1000,-2000,-5000",
                    help="Comma-separated time offsets to try for missing views")
    ap.add_argument("--out_dir", default="/simurgh2/datasets/HOI-M3/sam3_xv_probe")
    ap.add_argument("--img_root", default="/simurgh2/datasets/HOI-M3/images")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_subdir = os.path.join(args.out_dir, f"{args.seq}__{args.obj}__refv{args.ref_view}_f{args.ref_frame}")
    os.makedirs(out_subdir, exist_ok=True)

    # 1) sort views by camera angular position
    centers = load_calib_for_seq(args.seq)
    sorted_views = sort_views_by_angle(centers, args.ref_view)
    print(f"[xv] view order (starting from ref_view {args.ref_view}): {sorted_views[:10]}...")

    # 2) materialize "video"
    tmp_dir, frame_to_view = build_video(args.seq, args.ref_frame, sorted_views, args.img_root)
    ref_frame_idx = 0  # ref_view is first in sorted order

    # 3a) First: SAM3 image on ref_view to get bbox of correct detection
    sys.path.insert(0, "/simurgh/u/juze/code/sam3")
    from PIL import Image
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    print("[xv] loading SAM3 image model (for ref bbox)...")
    img_model = build_sam3_image_model(load_from_HF=True)
    img_proc = Sam3Processor(img_model)
    ref_img_path = os.path.join(args.img_root, args.seq, str(args.ref_view), f"{args.ref_frame:06d}.jpg")
    pil = Image.open(ref_img_path).convert("RGB")
    state = img_proc.set_image(pil)
    img_out = img_proc.set_text_prompt(state=state, prompt=args.obj)
    boxes = img_out.get("boxes")
    scores = img_out.get("scores")
    if hasattr(boxes, "cpu"): boxes = boxes.cpu().numpy()
    if hasattr(scores, "cpu"): scores = scores.cpu().numpy()
    boxes = np.asarray(boxes); scores = np.asarray(scores)
    if len(scores) == 0:
        print(f"[xv] ERR: no detection of '{args.obj}' in ref view {args.ref_view}")
        return 1
    print(f"[xv] ref view {args.ref_view} detections: {len(scores)}")
    for i, (b, s) in enumerate(zip(boxes, scores)):
        marker = " <-- using" if i == args.detection_idx else ""
        print(f"    [{i}] bbox={b.tolist()} score={float(s):.3f}{marker}")
    if args.detection_idx >= len(boxes):
        print(f"[xv] ERR: detection_idx {args.detection_idx} out of range")
        return 1
    chosen_box = boxes[args.detection_idx]  # xyxy

    # Convert xyxy (pixel) to xywh, then normalize to [0,1]
    img_w, img_h = pil.size
    x1, y1, x2, y2 = chosen_box.astype(float)
    bbox_xywh = [x1 / img_w, y1 / img_h, (x2 - x1) / img_w, (y2 - y1) / img_h]
    print(f"[xv] normalized bbox (xywh, 0-1): {bbox_xywh}")

    # Free img model
    del img_model, img_proc
    import torch
    torch.cuda.empty_cache()

    # 3b) SAM3 video propagation with BBOX prompt
    from sam3.model_builder import build_sam3_video_predictor
    print("[xv] loading SAM3 video predictor...")
    predictor = build_sam3_video_predictor()
    print("[xv] start session")
    start_resp = predictor.handle_request(dict(type="start_session", resource_path=tmp_dir))
    session_id = start_resp["session_id"]
    print(f"[xv] session={session_id} num_frames={start_resp.get('num_frames', '?')}")

    print(f"[xv] add BBOX prompt at frame {ref_frame_idx} (view {args.ref_view}): {bbox_xywh}")
    prompt_resp = predictor.handle_request(dict(
        type="add_prompt", session_id=session_id,
        frame_index=ref_frame_idx,
        text=args.obj,
        bounding_boxes=[bbox_xywh],
        bounding_box_labels=[1],
    ))
    init_output = prompt_resp.get("outputs", {})
    print(f"[xv] init detection on ref view: {list(init_output.keys()) if isinstance(init_output, dict) else type(init_output)}")

    # 4) propagate
    print("[xv] propagate across all 42 views...")
    outputs_per_frame = {}
    for stream in predictor.handle_stream_request(dict(type="propagate_in_video", session_id=session_id)):
        fidx = stream["frame_index"]
        outputs_per_frame[fidx] = stream["outputs"]

    print(f"[xv] propagation done — {len(outputs_per_frame)} frames")

    # 5) save per-view mask + overlay preview
    per_view_masks = {}
    # Debug: print structure of first frame output
    if outputs_per_frame:
        first_fidx = next(iter(outputs_per_frame))
        first_out = outputs_per_frame[first_fidx]
        print(f"[xv] DEBUG output[{first_fidx}] type: {type(first_out)}")
        if isinstance(first_out, dict):
            for k, v in first_out.items():
                t = type(v).__name__
                shape = getattr(v, 'shape', None) or (len(v) if hasattr(v, '__len__') else None)
                print(f"  '{k}': type={t}, shape/len={shape}")
    # Determine the bbox-prompted obj_id from init_output (the prompted frame).
    # When we add a bbox+text prompt, SAM3 returns multiple objects (bbox-prompted +
    # text-detected). We want the bbox one — pick the obj whose mask in init covers
    # the chosen bbox centroid.
    init_bms = init_output.get("out_binary_masks")
    init_ids = init_output.get("out_obj_ids")
    if hasattr(init_bms, "cpu"): init_bms = init_bms.cpu().numpy()
    if hasattr(init_ids, "cpu"): init_ids = init_ids.cpu().numpy()
    init_bms = np.asarray(init_bms)
    init_ids = np.asarray(init_ids)
    print(f"[xv] init obj_ids: {init_ids.tolist() if init_ids.size else 'none'}")
    target_obj_id = None
    if init_bms.size > 0 and init_ids.size > 0:
        ih, iw = init_bms.shape[-2:]
        # Centroid of our bbox in pixel coords
        cx = (x1 + x2) / 2 * (iw / img_w)
        cy = (y1 + y2) / 2 * (ih / img_h)
        cx_i, cy_i = int(cx), int(cy)
        for i, oid in enumerate(init_ids):
            mask = init_bms[i]
            if 0 <= cy_i < mask.shape[0] and 0 <= cx_i < mask.shape[1] and mask[cy_i, cx_i]:
                target_obj_id = int(oid)
                print(f"[xv] target obj_id = {target_obj_id} (covers bbox centroid)")
                break
    if target_obj_id is None:
        target_obj_id = int(init_ids[0]) if init_ids.size > 0 else 0
        print(f"[xv] fallback target obj_id = {target_obj_id}")

    for fidx, out in outputs_per_frame.items():
        view_id = frame_to_view[fidx]
        bms = out.get("out_binary_masks", [])
        ids = out.get("out_obj_ids", [])
        if hasattr(bms, "cpu"): bms = bms.cpu().numpy()
        if hasattr(ids, "cpu"): ids = ids.cpu().numpy()
        bms = np.asarray(bms); ids = np.asarray(ids)
        if bms.size == 0 or ids.size == 0:
            continue
        # Find the target_obj_id in this frame's detections
        match = np.where(ids == target_obj_id)[0]
        if len(match) == 0:
            continue
        m = bms[match[0]].astype(np.uint8)
        if m.sum() > 0:
            per_view_masks[view_id] = m

    # 5b) Fallback: for views with no propagated mask, scan adjacent frames
    # using SAM3 image text prompt. Pick first detection ≥ score threshold
    # whose bbox is roughly the same SIZE as the ref bbox (sanity filter).
    missing_views = [v for v in range(N_VIEWS) if v not in per_view_masks]
    print(f"\n[xv] views with NO propagation mask: {missing_views}")

    if missing_views:
        # Need SAM3 image model back
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        print("[xv] loading SAM3 image model for fallback...")
        img_model_fb = build_sam3_image_model(load_from_HF=True)
        img_proc_fb = Sam3Processor(img_model_fb)

        # Expected bbox size in pixels (from ref view) — use as size sanity
        ref_box_w = x2 - x1
        ref_box_h = y2 - y1
        ref_box_area = ref_box_w * ref_box_h
        print(f"[xv] ref bbox area={ref_box_area:.0f} (size filter: 0.2x ~ 5x)")

        offsets = [int(s) for s in args.fallback_frame_offsets.split(",")]
        per_view_ref_frame = {v: args.ref_frame for v in per_view_masks}

        for v in missing_views:
            found = False
            for off in offsets:
                f_try = args.ref_frame + off
                if f_try < 0:
                    continue
                ip = os.path.join(args.img_root, args.seq, str(v), f"{f_try:06d}.jpg")
                if not os.path.isfile(ip):
                    continue
                try:
                    pil_try = Image.open(ip).convert("RGB")
                    state_try = img_proc_fb.set_image(pil_try)
                    out_try = img_proc_fb.set_text_prompt(state=state_try, prompt=args.obj)
                    boxes_try = out_try.get("boxes")
                    scores_try = out_try.get("scores")
                    masks_try = out_try.get("masks")
                    if hasattr(boxes_try, "cpu"): boxes_try = boxes_try.cpu().numpy()
                    if hasattr(scores_try, "cpu"): scores_try = scores_try.cpu().numpy()
                    if hasattr(masks_try, "cpu"): masks_try = masks_try.cpu().numpy()
                    if len(scores_try) == 0:
                        continue
                    # Filter: score threshold + size sanity
                    candidates = []
                    for i, (b, s) in enumerate(zip(boxes_try, scores_try)):
                        if s < args.fallback_score_thresh:
                            continue
                        bw, bh = b[2] - b[0], b[3] - b[1]
                        area = bw * bh
                        # 0.2x ~ 5x of ref box area
                        if not (0.2 * ref_box_area <= area <= 5.0 * ref_box_area):
                            continue
                        candidates.append((s, i, area))
                    if not candidates:
                        continue
                    # Pick highest score
                    candidates.sort(reverse=True)
                    _, best_i, _ = candidates[0]
                    m_try = np.asarray(masks_try[best_i])
                    while m_try.ndim > 2:
                        m_try = m_try[0]
                    if m_try.sum() > 100:
                        per_view_masks[v] = m_try.astype(np.uint8)
                        per_view_ref_frame[v] = f_try
                        print(f"  [v{v:02d}] fallback found at frame {f_try} "
                              f"(offset={off:+d}, score={scores_try[best_i]:.2f})")
                        found = True
                        break
                except Exception as e:
                    print(f"  [v{v:02d}] f{f_try} ERR: {e}")
            if not found:
                print(f"  [v{v:02d}] NO fallback detection across {len(offsets)} offsets")

        del img_model_fb, img_proc_fb
        torch.cuda.empty_cache()
    else:
        per_view_ref_frame = {v: args.ref_frame for v in per_view_masks}

    # save masks as compressed npz + per-view ref frame
    npz_path = os.path.join(out_subdir, "per_view_masks.npz")
    save_dict = {f"v{v}": per_view_masks[v].astype(np.uint8) for v in per_view_masks}
    np.savez_compressed(npz_path, **save_dict)
    # per-view ref frame
    rf_path = os.path.join(out_subdir, "per_view_ref_frame.json")
    json.dump({str(v): int(per_view_ref_frame[v]) for v in per_view_ref_frame}, open(rf_path, "w"))
    print(f"[xv] saved {len(per_view_masks)} masks → {npz_path}")
    print(f"[xv] saved per-view ref frames → {rf_path}")

    # 6) build 6x7 grid overlay preview — show per-view ref frame's image with mask
    GRID_ROWS, GRID_COLS = 6, 7
    CELL_W, CELL_H = 320, 180
    grid = np.zeros((GRID_ROWS * CELL_H, GRID_COLS * CELL_W, 3), dtype=np.uint8)
    for v in range(N_VIEWS):
        # Use this view's ref frame (anchor frame OR fallback frame)
        f_show = per_view_ref_frame.get(v, args.ref_frame)
        ip = os.path.join(args.img_root, args.seq, str(v), f"{f_show:06d}.jpg")
        if os.path.isfile(ip):
            img = cv2.imread(ip)
        else:
            img = np.zeros((720, 1280, 3), dtype=np.uint8)
        if v in per_view_masks:
            m = per_view_masks[v]
            if m.shape != img.shape[:2]:
                m = cv2.resize(m.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            sel = m.astype(bool)
            if sel.sum() > 0:
                overlay = img.copy().astype(np.float32)
                overlay[sel] = overlay[sel] * 0.5 + np.array([0, 255, 0], dtype=np.float32) * 0.5
                img = overlay.clip(0, 255).astype(np.uint8)
                src = "OK" if f_show == args.ref_frame else f"fb_f{f_show}"
                cv2.putText(img, src, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        is_ref = (v == args.ref_view)
        col = (0, 255, 255) if is_ref else (255, 255, 255)
        cv2.putText(img, f"v{v:02d}" + (" REF" if is_ref else ""), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
        cell = cv2.resize(img, (CELL_W, CELL_H), interpolation=cv2.INTER_AREA)
        r, c = divmod(v, GRID_COLS)
        grid[r*CELL_H:(r+1)*CELL_H, c*CELL_W:(c+1)*CELL_W] = cell
    grid_path = os.path.join(out_subdir, "grid_preview.png")
    cv2.imwrite(grid_path, grid)
    print(f"[xv] grid preview → {grid_path}")

    # cleanup tmp video
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("[xv] done")


if __name__ == "__main__":
    main()
