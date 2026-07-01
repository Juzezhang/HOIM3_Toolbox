#!/usr/bin/env python3
"""Step 1: Convert mask shards + video frames → WebDataset .tar shards for YOLO-seg training.

Also writes a YOLO-format validation subset and class mapping / data.yaml files.

Usage:
    python scripts/yolo_seg/prepare_webdataset.py --step 60 --imgsz 640 --workers 8
    python scripts/yolo_seg/prepare_webdataset.py --sequences bedroom_data01 bedroom_data02
"""

import argparse
import json
import os
import sys
from os.path import join

import subprocess
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# ── Project root on sys.path so cross-package imports work ────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

import webdataset as wds  # noqa: E402

from scripts.utils.mask_io import SequenceShardReaders, load_frame_masks_shard_full  # noqa: E402
from scripts.visualize_mask_validity import preload_video_frames, resolve_view_validity  # noqa: E402
from scripts.yolo_seg.config import (  # noqa: E402
    SHARD_ROOT, VALIDITY_ROOT, VIDEO_ROOT, WDS_ROOT,
    MASK_H, MASK_W, NUM_VIEWS, FRAME_STEP, IMGSZ, SAMPLES_PER_SHARD,
    correct_name, load_sequence_contents, build_class_mapping, get_train_sequences,
)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Prepare WebDataset shards for YOLO-seg training")
    p.add_argument("--step", type=int, default=FRAME_STEP,
                   help="Frame step (default: 60 ≈ 1 fps)")
    p.add_argument("--imgsz", type=int, default=IMGSZ,
                   help="Image width; height auto-computed for 16:9")
    p.add_argument("--workers", type=int, default=8,
                   help="Threads for parallel video I/O")
    p.add_argument("--sequences", nargs="+", default=None,
                   help="Optional subset of sequences to process")
    p.add_argument("--val_step", type=int, default=300,
                   help="Frame step for validation subset")
    p.add_argument("--val_every", type=int, default=5,
                   help="Take every Nth training sequence for validation")
    p.add_argument("--jpeg_quality", type=int, default=95,
                   help="JPEG quality for shard images")
    p.add_argument("--views", type=int, nargs="+", default=None,
                   help="Subset of views to process (default: all 42)")
    p.add_argument("--output", type=str, default=None,
                   help="Override output root (default: WDS_ROOT from config)")
    return p.parse_args()


# ── Annotation extraction ─────────────────────────────────────────────────────

def mask_to_bbox_polygon(mask_binary: np.ndarray, min_contour_area: int = 100):
    """Extract normalised bbox + polygon(s) from a binary mask.

    Returns:
        bbox: [cx, cy, w, h] normalised to [0, 1], or None
        segments: list of polygons, each [[x1,y1], ...] normalised, or None
    """
    if not np.any(mask_binary):
        return None, None

    mask_u8 = mask_binary.astype(np.uint8) * 255
    x, y, w, h = cv2.boundingRect(mask_u8)
    if w == 0 or h == 0:
        return None, None

    img_h, img_w = mask_binary.shape
    bbox = [
        (x + w / 2.0) / img_w,
        (y + h / 2.0) / img_h,
        w / img_w,
        h / img_h,
    ]

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    segments = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_contour_area:
            continue
        pts = cnt.reshape(-1, 2).astype(np.float64)
        pts[:, 0] /= img_w
        pts[:, 1] /= img_h
        segments.append(pts.tolist())

    if not segments:
        return None, None
    return bbox, segments


# ── Video frame helpers ───────────────────────────────────────────────────────

def _encode_jpeg(frame: np.ndarray, quality: int = 95) -> bytes | None:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None


def _preload_view_jpegs(video_path, frame_indices, target_size, quality):
    """Extract specific frames from video, auto-selecting the fastest method.

    - Few frames (≤200): OpenCV seeking — seeks to each frame directly.
    - Many frames (>200): ffmpeg streaming — decodes all, picks targets.
    """
    if len(frame_indices) <= 200:
        return _preload_view_jpegs_seek(video_path, frame_indices, target_size, quality)
    return _preload_view_jpegs_stream(video_path, frame_indices, target_size, quality)


def _preload_view_jpegs_seek(video_path, frame_indices, target_size, quality):
    """OpenCV seeking — fast when only a few frames are needed (e.g. step=600)."""
    target_w, target_h = target_size
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}
    results: dict[int, bytes] = {}
    for idx in sorted(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (target_w, target_h))
        jpeg = _encode_jpeg(frame, quality)
        if jpeg is not None:
            results[idx] = jpeg
    cap.release()
    return results


def _preload_view_jpegs_stream(video_path, frame_indices, target_size, quality):
    """ffmpeg streaming — fast when many frames are needed (e.g. step=60)."""
    target_w, target_h = target_size
    frame_set = set(frame_indices)

    cmd = [
        "ffmpeg",
        "-skip_frame", "nokey",
        "-flags2", "+fast",
        "-i", video_path,
        "-vf", f"scale={target_w}:{target_h}",
        "-pix_fmt", "bgr24",
        "-f", "rawvideo",
        "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    frame_size = target_w * target_h * 3
    results: dict[int, bytes] = {}
    idx = 0

    while True:
        raw = proc.stdout.read(frame_size)
        if len(raw) < frame_size:
            break
        if idx in frame_set:
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(target_h, target_w, 3)
            jpeg = _encode_jpeg(frame.copy(), quality)
            if jpeg is not None:
                results[idx] = jpeg
        idx += 1

    proc.stdout.close()
    proc.wait()
    return results


# ── YOLO-format validation writer ────────────────────────────────────────────

class YOLOValWriter:
    """Writes samples in standard YOLO directory layout (images/ + labels/)."""

    def __init__(self, val_dir: str):
        self.img_dir = join(val_dir, "images")
        self.lbl_dir = join(val_dir, "labels")
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.lbl_dir, exist_ok=True)

    def write(self, key: str, jpeg_bytes: bytes, annotations: list):
        # Image
        img_path = join(self.img_dir, f"{key}.jpg")
        with open(img_path, "wb") as f:
            f.write(jpeg_bytes)

        # Label: YOLO segment format → class_id px1 py1 px2 py2 …
        # (no explicit bbox — YOLO auto-computes it from the polygon)
        # One line per instance, using the largest contour.
        lbl_path = join(self.lbl_dir, f"{key}.txt")
        lines = []
        for ann in annotations:
            # Pick the contour with the most points
            largest_seg = max(ann["segments"], key=len)
            parts = [str(ann["class_id"])]
            for pt in largest_seg:
                parts.extend(f"{v:.6f}" for v in pt)
            lines.append(" ".join(parts))
        with open(lbl_path, "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))


# ── Per-sequence processing ───────────────────────────────────────────────────

def process_sequence(
    seq_name: str,
    class_to_id: dict,
    shard_writer,
    args,
    val_writer: YOLOValWriter | None = None,
) -> int:
    """Process one sequence: read masks + frames → write WebDataset samples.

    Returns the number of samples written.
    """
    shard_path = join(SHARD_ROOT, seq_name)
    validity_dir = join(VALIDITY_ROOT, seq_name)
    videos_dir = join(VIDEO_ROOT, seq_name, "videos")

    meta_file = join(shard_path, "meta.json")
    if not os.path.isdir(shard_path) or not os.path.isfile(meta_file):
        print(f"  Skip {seq_name}: no completed shard (missing meta.json)", flush=True)
        return 0

    seq_readers = SequenceShardReaders(shard_path)
    frame_ids = seq_readers.frame_ids_list

    target_frames = frame_ids[:: args.step]
    if not target_frames:
        seq_readers.close()
        return 0

    # Object name → YOLO class id
    obj_class_map: dict[str, int] = {}
    for obj in seq_readers.objects:
        clean = correct_name(obj)
        if clean.startswith("person"):
            obj_class_map[obj] = class_to_id["person"]
        elif clean in class_to_id:
            obj_class_map[obj] = class_to_id[clean]
        else:
            print(f"  Warning: unknown object '{obj}' (→ '{clean}'), skipping", flush=True)

    # Image target size (16:9)
    tgt_w = args.imgsz
    tgt_h = int(tgt_w * 9 / 16)  # 640 → 360
    tgt_size = (tgt_w, tgt_h)

    # Validation frame set (subset of target_frames)
    val_frame_set: set[int] = set()
    if val_writer is not None:
        val_frame_set = set(frame_ids[:: args.val_step])

    # ── Preload video frames as JPEG bytes (threaded across views) ────────
    view_list = args.views if args.views is not None else list(range(NUM_VIEWS))
    print(f"  Preloading {len(target_frames)} frames × {len(view_list)} views …", flush=True)
    view_jpegs: dict[int, dict[int, bytes]] = {}

    def _load_one_view(v):
        vpath = join(videos_dir, f"{v}.mp4")
        if not os.path.exists(vpath):
            return v, {}
        return v, _preload_view_jpegs(vpath, target_frames, tgt_size, args.jpeg_quality)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for v, jpegs in pool.map(lambda v: _load_one_view(v), view_list):
            if jpegs:
                view_jpegs[v] = jpegs

    print(f"  Loaded frames for {len(view_jpegs)}/{len(view_list)} views", flush=True)

    # ── Iterate frames ────────────────────────────────────────────────────
    all_views = list(range(NUM_VIEWS))  # used for validity resolution
    sample_count = 0

    for frame_id in tqdm(target_frames, desc=f"  {seq_name}", leave=False):
        # Load masks (all objects, all views)
        try:
            masks = load_frame_masks_shard_full(seq_readers, frame_id)
        except KeyError:
            continue

        # Load validity
        validity: dict[str, np.ndarray] = {}
        vfile = join(validity_dir, f"{frame_id:06d}.npz")
        if os.path.exists(vfile):
            with np.load(vfile) as vdata:
                validity = {k: vdata[k] for k in vdata.files}

        for view_idx in view_list:
            if view_idx not in view_jpegs:
                continue
            jpeg = view_jpegs[view_idx].get(frame_id)
            if jpeg is None:
                continue

            # Build annotations for this (frame, view) pair
            annotations: list[dict] = []
            for obj_name, cls_id in obj_class_map.items():
                # Validity check
                vkey = f"{obj_name}_validity"
                if vkey in validity:
                    if not resolve_view_validity(validity[vkey], view_idx, view_idx, all_views):
                        continue

                if obj_name not in masks:
                    continue
                mask_view = masks[obj_name][view_idx]
                bbox, segments = mask_to_bbox_polygon(mask_view > 127)
                if bbox is None:
                    continue

                annotations.append({
                    "class_id": cls_id,
                    "bbox": bbox,
                    "segments": segments,
                })

            if not annotations:
                continue

            key = f"{seq_name}_{frame_id:06d}_{view_idx:02d}"

            # Write to WebDataset shard
            ann_bytes = json.dumps({
                "img_w": MASK_W, "img_h": MASK_H,
                "annotations": annotations,
            }).encode("utf-8")
            shard_writer.write({"__key__": key, "jpg": jpeg, "json": ann_bytes})
            sample_count += 1

            # Write to YOLO val set if applicable
            if frame_id in val_frame_set and val_writer is not None:
                val_writer.write(key, jpeg, annotations)

    seq_readers.close()
    return sample_count


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    seq_contents = load_sequence_contents()
    class_to_id, class_names = build_class_mapping(seq_contents)
    print(f"Classes: {len(class_names)} (person + {len(class_names) - 1} objects)")

    train_seqs = get_train_sequences(seq_contents)
    if args.sequences:
        allowed = set(args.sequences)
        train_seqs = [s for s in train_seqs if s in allowed]
    print(f"Training sequences: {len(train_seqs)}")

    # ── Output directories & metadata ─────────────────────────────────────
    output_root = args.output if args.output else WDS_ROOT
    shard_dir = join(output_root, "train_shards")
    val_dir = join(output_root, "val")
    os.makedirs(shard_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Class mapping JSON
    with open(join(output_root, "class_mapping.json"), "w") as f:
        json.dump({"class_to_id": class_to_id, "class_names": class_names}, f, indent=2)

    # YOLO data.yaml
    with open(join(output_root, "data.yaml"), "w") as f:
        f.write(f"path: {output_root}\n")
        f.write("train: train_shards\n")
        f.write("val: val/images\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write("names:\n")
        for i, name in enumerate(class_names):
            f.write(f"  {i}: {name}\n")

    print(f"Wrote class_mapping.json + data.yaml → {WDS_ROOT}")

    # ── Validation sequence selection ─────────────────────────────────────
    val_seq_set = set(train_seqs[:: args.val_every])
    val_writer = YOLOValWriter(val_dir)
    print(f"Validation sequences: {len(val_seq_set)} (every {args.val_every}th)")

    # ── Shard writer ──────────────────────────────────────────────────────
    pattern = join(shard_dir, "shard-%06d.tar")
    writer = wds.ShardWriter(pattern, maxcount=SAMPLES_PER_SHARD)

    total = 0
    for seq in train_seqs:
        print(f"\n{'='*60}\n{seq}", flush=True)
        n = process_sequence(
            seq, class_to_id, writer, args,
            val_writer=val_writer if seq in val_seq_set else None,
        )
        total += n
        print(f"  → {n} samples (cumulative {total})", flush=True)

    writer.close()
    print(f"\nDone. {total} samples across shards in {shard_dir}")
    print(f"Validation images/labels in {val_dir}")


if __name__ == "__main__":
    main()
