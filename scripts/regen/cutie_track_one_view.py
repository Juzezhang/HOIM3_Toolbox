#!/usr/bin/env python3
"""Step 2 of Cutie mask-tracking pipeline: track ONE (seq, view) at 1080p.

Output layout
-------------
Per-frame compressed npz at::

    <output_root>/<seq>/<view>/<frame:06d>.npz
        keys: "mask" -> (1080, 1920) uint8 INDEXED mask (0=bg, 1..N=objects)
              "names" -> JSON-encoded list of object names (length N+1, [bg, obj1, ...])

Choice of layout (per-frame indexed npz):
- Per-frame because we want to checkpoint partial progress (Cutie can be slow).
- Indexed uint8 because compresses very well (mostly bg=0 in any single view) —
  typical compressed file size ~30 KB/frame vs ~50 GB for a single uncompressed
  (n_frames, 1080, 1920) per view.
- Names embedded so aggregator (Step 3) doesn't need cross-view JSON sidecars.

Sentinel: ``<output_root>/<seq>/<view>/.tracked_done`` (with frame count).

Cutie is trimmed from ``Trackingmask/demo.py``. Key differences:
- Single-seq single-view CLI (no Hydra dataset config).
- Uses ``max_internal_size=480`` so Cutie runs internally at 480p (output upsampled).
- Skips ``gui.resource_manager_zhangjy.ResourceManager`` (no visualization, no DAVIS).
- Loads the (1080, 1920) indexed reference from Step 1 (cutie_refs dir).

Usage
-----
    python cutie_track_one_view.py \\
        --seq office_data32 --view 0 \\
        --ref_root /simurgh2/datasets/HOI-M3/cutie_refs \\
        --video_root /simurgh2/datasets/HOI-M3/videos \\
        --output_root /simurgh2/datasets/HOI-M3/cutie_tracking
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from os.path import join

import cv2
import numpy as np

# Ensure we run from Trackingmask dir so Hydra finds cutie/config
TRACKING_DIR = "/simurgh/u/juze/code/HHOI-Toolkit/Trackingmask"
sys.path.insert(0, TRACKING_DIR)

TARGET_W = 1920
TARGET_H = 1080

# Per-seq overrides for ref_root / output_root.
# Used to piggyback bedroom_data01 (Cutie re-tracking from non-zero start frames)
# on the same dispatcher / worker loop without disturbing the 11 office seqs.
_REF_ROOT_OVERRIDES = {
    "bedroom_data01": "/simurgh2/datasets/HOI-M3/cutie_refs_bedroom_data01",
}
_OUTPUT_ROOT_OVERRIDES = {
    "bedroom_data01": "/scr/juze/datasets/HOI-M3/cutie_tracking_bedroom_data01",
}


def _save_frame(out_dir, frame_idx, indexed_mask, names):
    """Save per-frame compressed npz (atomic via tmp rename)."""
    path = join(out_dir, f"{frame_idx:06d}.npz")
    tmp_base = join(out_dir, f"{frame_idx:06d}.tmp")
    # np.savez_compressed auto-appends .npz to the given filename.
    np.savez_compressed(tmp_base, mask=indexed_mask, names=np.array(names))
    os.replace(tmp_base + ".npz", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--view", type=int, required=True)
    ap.add_argument("--ref_root", default="/simurgh2/datasets/HOI-M3/cutie_refs")
    ap.add_argument("--video_root", default="/simurgh2/datasets/HOI-M3/videos")
    ap.add_argument(
        "--output_root", default="/simurgh2/datasets/HOI-M3/cutie_tracking"
    )
    ap.add_argument("--max_internal_size", type=int, default=480)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mem_every", type=int, default=5)
    ap.add_argument("--frame_stride", type=int, default=1,
                    help="Cutie tracks every Nth frame; skipped frames copy mask from last tracked. "
                         "2 → 2x speedup, ~equivalent quality for slow-moving objects.")
    ap.add_argument(
        "--max_frames",
        type=int,
        default=-1,
        help="Cap on number of frames (debug/smoketest). -1 = all.",
    )
    ap.add_argument(
        "--start_frame",
        type=int,
        default=0,
        help=(
            "Frame index used as the Cutie reference (mask provided). Frames "
            "before this are SKIPPED (no output). If "
            "<ref_root_seq>/masks/<view>_start_frame.txt exists, its value "
            "overrides this CLI default."
        ),
    )
    ap.add_argument(
        "--end_frame",
        type=int,
        default=-1,
        help=(
            "Exclusive end frame for the FORWARD pass (-1 = track to video "
            "end). Used for mid-sequence hole fills (added 2026-06-09)."
        ),
    )
    ap.add_argument(
        "--forward_only",
        action="store_true",
        help=(
            "Skip the backward pass (start_frame-1 -> 0) even when "
            "start_frame > 0. Used for mid-sequence hole fills where frames "
            "before start_frame are already valid (added 2026-06-09)."
        ),
    )
    args = ap.parse_args()

    seq = args.seq
    view = args.view

    # Per-seq override for ref_root / output_root (e.g. bedroom_data01 piggyback
    # without colliding with the running office seqs).
    ref_root = _REF_ROOT_OVERRIDES.get(seq, args.ref_root)
    output_root = _OUTPUT_ROOT_OVERRIDES.get(seq, args.output_root)
    if ref_root != args.ref_root:
        print(f"[{seq}/v{view}] ref_root OVERRIDE: {ref_root}", flush=True)
    if output_root != args.output_root:
        print(f"[{seq}/v{view}] output_root OVERRIDE: {output_root}", flush=True)

    mask_path = join(ref_root, seq, "masks", f"{view}.npy")
    names_path = join(ref_root, seq, "masks", f"{view}_names.json")
    start_frame_path = join(ref_root, seq, "masks", f"{view}_start_frame.txt")
    video_path = join(args.video_root, seq, "videos", f"{view}.mp4")
    out_dir = join(output_root, seq, str(view))
    sentinel = join(out_dir, ".tracked_done")

    # Resolve start_frame: file overrides CLI default, CLI default overrides 0.
    start_frame = args.start_frame
    if os.path.isfile(start_frame_path):
        with open(start_frame_path) as f:
            start_frame = int(f.read().strip())
        print(
            f"[{seq}/v{view}] start_frame={start_frame} (from {start_frame_path})",
            flush=True,
        )
    elif start_frame != 0:
        print(f"[{seq}/v{view}] start_frame={start_frame} (CLI)", flush=True)

    if os.path.isfile(sentinel):
        print(f"SKIP {seq}/v{view}: sentinel exists ({sentinel})", flush=True)
        return 0

    if not os.path.isfile(mask_path):
        print(f"FAIL {seq}/v{view}: no reference mask at {mask_path}", flush=True)
        return 2
    if not os.path.isfile(video_path):
        print(f"FAIL {seq}/v{view}: no video at {video_path}", flush=True)
        return 2

    os.makedirs(out_dir, exist_ok=True)

    # Lazy imports (after sys.path tweak)
    import torch  # noqa
    from hydra import compose, initialize_config_dir  # noqa
    from omegaconf import open_dict  # noqa
    from cutie.inference.inference_core import InferenceCore  # noqa
    from cutie.model.cutie import CUTIE  # noqa
    from gui.interactive_utils import (  # noqa
        image_to_torch,
        torch_prob_to_numpy_mask,
        index_numpy_to_one_hot_torch,
    )

    # Build Hydra config using absolute path (initialize_config_dir doesn't depend on cwd).
    torch.cuda.empty_cache()
    config_dir_abs = join(TRACKING_DIR, "cutie", "config")
    from hydra.core.global_hydra import GlobalHydra
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with torch.inference_mode():
        initialize_config_dir(
            version_base="1.3.2",
            config_dir=config_dir_abs,
            job_name="eval_config",
        )
        cfg = compose(config_name="eval_config")
        GlobalHydra.instance().clear()
    with open_dict(cfg):
        cfg["weights"] = join(TRACKING_DIR, "weights/cutie-base-mega.pth")
        cfg["max_internal_size"] = args.max_internal_size
        cfg["mem_every"] = args.mem_every
        cfg["amp"] = True

    print(
        f"[{seq}/v{view}] cfg: max_internal_size={cfg.max_internal_size} "
        f"mem_every={cfg.mem_every} amp={cfg.amp}",
        flush=True,
    )

    # Load model (a fresh InferenceCore is created per pass below)
    cutie = CUTIE(cfg).to(args.device).eval()
    weights = torch.load(cfg.weights, map_location=args.device)
    cutie.load_weights(weights)

    # Load reference indexed mask & names
    template_idx = np.load(mask_path)  # (1080, 1920) uint8
    if template_idx.shape != (TARGET_H, TARGET_W):
        print(
            f"WARN {seq}/v{view}: template shape {template_idx.shape} != "
            f"({TARGET_H},{TARGET_W}); resizing",
            flush=True,
        )
        template_idx = cv2.resize(
            template_idx, (TARGET_W, TARGET_H), interpolation=cv2.INTER_NEAREST
        )
    with open(names_path) as f:
        names_data = json.load(f)
    mask_names = names_data["mask_names"]  # ordered list of object names
    # Full names including background as index 0
    full_names = ["background"] + list(mask_names)

    # Number of objects = max ID in template OR len(mask_names)
    num_objects = len(mask_names)
    # Sanity: any IDs in template > num_objects?
    max_id = int(template_idx.max())
    if max_id > num_objects:
        print(
            f"WARN {seq}/v{view}: template max_id={max_id} > num_objects={num_objects}",
            flush=True,
        )

    # Probe video for total_frames; per-pass captures are opened inside _run_pass.
    _probe = cv2.VideoCapture(video_path)
    if not _probe.isOpened():
        print(f"FAIL {seq}/v{view}: cannot open {video_path}", flush=True)
        return 3
    total_frames = int(_probe.get(cv2.CAP_PROP_FRAME_COUNT))
    _probe.release()
    if args.max_frames > 0:
        total_frames = min(total_frames, args.max_frames)
    print(
        f"[{seq}/v{view}] total_frames={total_frames} objects={num_objects} ({mask_names})",
        flush=True,
    )
    end_frame = total_frames if args.end_frame < 0 else min(args.end_frame, total_frames)
    if start_frame < 0 or start_frame >= total_frames:
        print(
            f"FAIL {seq}/v{view}: start_frame={start_frame} out of range "
            f"[0, {total_frames})",
            flush=True,
        )
        return 4
    t0 = time.time()
    saved = 0

    # One-hot the template once: (num_classes, H, W) where num_classes = num_objects+1
    template_one_hot = index_numpy_to_one_hot_torch(
        template_idx.astype(np.int64), num_objects + 1
    )  # CPU float (C, H, W)
    # Drop background channel before feeding (per demo.py line 109 then 155)
    template_obj_only = template_one_hot[1:]  # (num_objects, H, W)
    template_obj_only_dev = template_obj_only.to(args.device)

    def _run_pass(direction):
        """Run Cutie either forward (start_frame -> total_frames-1) or backward
        (start_frame -> 0). Returns (n_saved, frames_fed).

        For each pass we open a fresh capture and reset the processor — the
        backward pass cannot share temporal memory with the forward pass.

        The reference frame (start_frame) is only saved during the FORWARD pass
        to avoid clobbering the indexed ref mask twice. Backward pass starts
        emitting at start_frame-1.
        """
        nonlocal saved
        local_t0 = time.time()
        last_log = local_t0
        # Fresh processor + capture per direction
        processor_local = InferenceCore(cutie, cfg=cfg)
        cap_local = cv2.VideoCapture(video_path)
        if not cap_local.isOpened():
            return 0, 0

        try:
            if direction == "forward":
                # Advance to start_frame
                for _ in range(start_frame):
                    if not cap_local.grab():
                        print(
                            f"FAIL {seq}/v{view}: failed to grab up to start_frame={start_frame}",
                            flush=True,
                        )
                        return 0, 0
                frame_indices = list(range(start_frame, end_frame))
                # Read each subsequent frame via cap.read()
                def _read_next(_state):
                    ret, fr = cap_local.read()
                    return ret, fr
                state = None
            else:
                # backward: need frames from 0 to start_frame in REVERSE order.
                # First, read frames 0..start_frame into memory at TARGET resolution
                # (downscaling first). This is bounded by start_frame which is
                # typically a few hundred frames (max scan cap = 5000).
                frames_buf = []
                for fi in range(start_frame + 1):
                    ret, fr = cap_local.read()
                    if not ret:
                        print(
                            f"FAIL {seq}/v{view}: backward buffer read failed at frame {fi}",
                            flush=True,
                        )
                        return 0, 0
                    fr = cv2.resize(fr, (TARGET_W, TARGET_H))
                    frames_buf.append(fr)
                # Iteration order: start_frame, start_frame-1, ..., 0
                frame_indices = list(range(start_frame, -1, -1))
                def _read_next(state):
                    j = state["i"]
                    if j >= len(frame_indices):
                        return False, None
                    fr = frames_buf[frame_indices[j]]
                    state["i"] += 1
                    return True, fr
                state = {"i": 0}

            step_count = 0
            n_saved_local = 0
            last_pred_idx = None  # for frame_stride > 1: copied to skipped frames
            for i_step, fi in enumerate(frame_indices):
                if direction == "forward":
                    ret, frame = _read_next(state)
                else:
                    ret, frame = _read_next(state)
                if not ret or frame is None:
                    break
                if direction == "forward":
                    frame = cv2.resize(frame, (TARGET_W, TARGET_H))

                # Skip already-saved frames (idempotency). For forward pass at
                # step 0 we still need to feed the template, so we always feed
                # but skip the write if file exists.
                already = os.path.isfile(join(out_dir, f"{fi:06d}.npz"))

                # frame_stride>1: Cutie tracks every Nth frame; in-between frames
                # copy mask from last tracked frame (no Cutie step → faster).
                # First frame (template) + start_frame always tracked.
                is_first_step = (step_count == 0)
                do_track = is_first_step or (i_step % args.frame_stride == 0)

                if do_track:
                    frame_torch = image_to_torch(frame, device=args.device)
                    if is_first_step:
                        prediction = processor_local.step(
                            frame_torch,
                            template_obj_only_dev,
                            idx_mask=False,
                        )
                    else:
                        prediction = processor_local.step(frame_torch)
                    pred_idx = torch_prob_to_numpy_mask(prediction)
                    if pred_idx.shape != (TARGET_H, TARGET_W):
                        pred_idx = cv2.resize(
                            pred_idx.astype(np.uint8),
                            (TARGET_W, TARGET_H),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    last_pred_idx = pred_idx
                # else: skipped frame, use last_pred_idx (copy from previous tracked)

                # Skip writing start_frame on backward pass (already written by
                # forward pass, and identical since both use the template).
                if direction == "backward" and fi == start_frame:
                    step_count += 1
                    continue

                if not already and last_pred_idx is not None:
                    _save_frame(out_dir, fi, last_pred_idx, full_names)
                    n_saved_local += 1

                step_count += 1

                now = time.time()
                if now - last_log > 30:
                    rate = step_count / max(now - local_t0, 1e-6)
                    if direction == "forward":
                        remain = end_frame - fi - 1
                    else:
                        remain = fi  # frames left to process backward
                    eta = remain / max(rate, 1e-6)
                    print(
                        f"  [{seq}/v{view}] {direction} {fi}/{total_frames} "
                        f"fps={rate:.1f} ETA={eta/60:.1f}min "
                        f"saved={n_saved_local}",
                        flush=True,
                    )
                    last_log = now
            return n_saved_local, step_count
        finally:
            cap_local.release()
            processor_local.clear_memory()
            processor_local.clear_non_permanent_memory()
            processor_local.clear_sensory_memory()
            torch.cuda.empty_cache()

    fwd_saved = 0
    bwd_saved = 0
    try:
        with torch.inference_mode():
            with torch.amp.autocast("cuda", enabled=bool(cfg.amp)):
                # Pass 1: forward (start_frame -> total_frames-1)
                fwd_saved, _ = _run_pass("forward")
                # Pass 2: backward (start_frame-1 -> 0)
                if start_frame > 0 and not args.forward_only:
                    bwd_saved, _ = _run_pass("backward")
    finally:
        torch.cuda.empty_cache()

    saved = fwd_saved + bwd_saved
    dt = time.time() - t0
    with open(sentinel, "w") as f:
        json.dump(
            {
                "seq": seq,
                "view": view,
                "frames_saved": saved,
                "frames_saved_forward": fwd_saved,
                "frames_saved_backward": bwd_saved,
                "total_frames": total_frames,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "forward_only": bool(args.forward_only),
                "wall_seconds": dt,
                "names": full_names,
                "shape": [TARGET_H, TARGET_W],
                "max_internal_size": args.max_internal_size,
            },
            f,
        )
    print(
        f"OK {seq}/v{view}: {saved} frames (fwd={fwd_saved} bwd={bwd_saved}) "
        f"in {dt:.0f}s ({saved/max(dt,1):.1f} fps)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
