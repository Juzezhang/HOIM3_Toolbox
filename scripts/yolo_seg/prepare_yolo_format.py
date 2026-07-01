#!/usr/bin/env python3
"""Generate YOLO standard directory format (images/ + labels/) from mask shards.

This bypasses WebDataset entirely and produces data that ultralytics can
train on natively with its built-in SegmentationDataset — no custom DataLoader needed.

Output:
    {output}/train/images/{seq}_{frame:06d}_{view:02d}.jpg
    {output}/train/labels/{seq}_{frame:06d}_{view:02d}.txt
    {output}/val/images/...
    {output}/val/labels/...
    {output}/data.yaml

Label format (YOLO segment): class_id px1 py1 px2 py2 ... pNx pNy

Usage:
    python scripts/yolo_seg/prepare_yolo_format.py --step 600 --imgsz 640 --workers 8
"""

import argparse
import json
import os
import subprocess
import sys
from os.path import join

import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

from scripts.utils.mask_io import SequenceShardReaders, load_frame_masks_shard_full
from scripts.visualize_mask_validity import preload_video_frames, resolve_view_validity
from scripts.yolo_seg.config import (
    SHARD_ROOT, VALIDITY_ROOT, VIDEO_ROOT,
    MASK_H, MASK_W, NUM_VIEWS, FRAME_STEP, IMGSZ,
    correct_name, load_sequence_contents, build_class_mapping, get_train_sequences,
)


def parse_args():
    p = argparse.ArgumentParser(description="Generate YOLO format training data")
    p.add_argument("--step", type=int, default=FRAME_STEP)
    p.add_argument("--imgsz", type=int, default=IMGSZ)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--sequences", nargs="+", default=None)
    p.add_argument("--views", type=int, nargs="+", default=None)
    p.add_argument("--val_step", type=int, default=None,
                   help="Frame step for val (default: step*5)")
    p.add_argument("--val_every", type=int, default=5)
    p.add_argument("--output", type=str, default="/simurgh2/datasets/HOI-M3/yolo_std")
    p.add_argument("--jpeg_quality", type=int, default=95)
    return p.parse_args()


def mask_to_polygon(mask_binary, min_contour_area=100):
    """Extract largest polygon from binary mask, normalized to [0,1]."""
    if not np.any(mask_binary):
        return None
    mask_u8 = mask_binary.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    if not contours:
        return None
    # Use largest contour
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_contour_area:
        return None
    img_h, img_w = mask_binary.shape
    pts = largest.reshape(-1, 2).astype(np.float64)
    pts[:, 0] /= img_w
    pts[:, 1] /= img_h
    return pts


def _preload_view_seek(video_path, frame_indices, target_size):
    """OpenCV seeking — for sparse frame extraction."""
    target_w, target_h = target_size
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}
    results = {}
    for idx in sorted(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (target_w, target_h))
        results[idx] = frame
    cap.release()
    return results


def process_sequence(seq_name, class_to_id, args, split, img_dir, lbl_dir):
    """Process one sequence, write images + labels in YOLO format."""
    shard_path = join(SHARD_ROOT, seq_name)
    validity_dir = join(VALIDITY_ROOT, seq_name)
    videos_dir = join(VIDEO_ROOT, seq_name, "videos")

    meta_file = join(shard_path, "meta.json")
    if not os.path.isfile(meta_file):
        return 0

    seq_readers = SequenceShardReaders(shard_path)
    frame_ids = seq_readers.frame_ids_list

    step = args.val_step if split == "val" else args.step
    target_frames = frame_ids[::step]
    if not target_frames:
        seq_readers.close()
        return 0

    # Object -> class mapping
    obj_class_map = {}
    for obj in seq_readers.objects:
        clean = correct_name(obj)
        if clean.startswith("person"):
            obj_class_map[obj] = class_to_id["person"]
        elif clean in class_to_id:
            obj_class_map[obj] = class_to_id[clean]

    tgt_w, tgt_h = args.imgsz, int(args.imgsz * 9 / 16)
    tgt_size = (tgt_w, tgt_h)
    view_list = args.views if args.views else list(range(NUM_VIEWS))

    # Preload video frames
    view_frames = {}
    def _load(v):
        vpath = join(videos_dir, f"{v}.mp4")
        if not os.path.exists(vpath):
            return v, {}
        return v, _preload_view_seek(vpath, target_frames, tgt_size)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for v, frames in pool.map(lambda v: _load(v), view_list):
            if frames:
                view_frames[v] = frames

    all_views = list(range(NUM_VIEWS))
    sample_count = 0

    for frame_id in tqdm(target_frames, desc=f"  {seq_name}", leave=False):
        try:
            masks = load_frame_masks_shard_full(seq_readers, frame_id)
        except KeyError:
            continue

        validity = {}
        vfile = join(validity_dir, f"{frame_id:06d}.npz")
        if os.path.exists(vfile):
            with np.load(vfile) as vdata:
                validity = {k: vdata[k] for k in vdata.files}

        for view_idx in view_list:
            if view_idx not in view_frames:
                continue
            frame = view_frames[view_idx].get(frame_id)
            if frame is None:
                continue

            # Build label lines
            label_lines = []
            for obj_name, cls_id in obj_class_map.items():
                vkey = f"{obj_name}_validity"
                if vkey in validity:
                    if not resolve_view_validity(validity[vkey], view_idx, view_idx, all_views):
                        continue
                if obj_name not in masks:
                    continue
                mask_view = masks[obj_name][view_idx]
                polygon = mask_to_polygon(mask_view > 127)
                if polygon is None:
                    continue
                # YOLO segment format: class_id px1 py1 px2 py2 ...
                parts = [str(cls_id)]
                for pt in polygon:
                    parts.extend(f"{v:.6f}" for v in pt)
                label_lines.append(" ".join(parts))

            if not label_lines:
                continue

            key = f"{seq_name}_{frame_id:06d}_{view_idx:02d}"
            # Write image
            img_path = join(img_dir, f"{key}.jpg")
            cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
            # Write label
            lbl_path = join(lbl_dir, f"{key}.txt")
            with open(lbl_path, "w") as f:
                f.write("\n".join(label_lines) + "\n")
            sample_count += 1

    seq_readers.close()
    return sample_count


def main():
    args = parse_args()
    if args.val_step is None:
        args.val_step = args.step * 5

    seq_contents = load_sequence_contents()
    class_to_id, class_names = build_class_mapping(seq_contents)
    print(f"Classes: {len(class_names)} (person + {len(class_names)-1} objects)")

    train_seqs = get_train_sequences(seq_contents)
    if args.sequences:
        allowed = set(args.sequences)
        train_seqs = [s for s in train_seqs if s in allowed]
    print(f"Training sequences: {len(train_seqs)}")

    # Output dirs
    train_img = join(args.output, "train", "images")
    train_lbl = join(args.output, "train", "labels")
    val_img = join(args.output, "val", "images")
    val_lbl = join(args.output, "val", "labels")
    for d in [train_img, train_lbl, val_img, val_lbl]:
        os.makedirs(d, exist_ok=True)

    # data.yaml
    yaml_path = join(args.output, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {args.output}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write("names:\n")
        for i, name in enumerate(class_names):
            f.write(f"  {i}: {name}\n")

    # Class mapping
    with open(join(args.output, "class_mapping.json"), "w") as f:
        json.dump({"class_to_id": class_to_id, "class_names": class_names}, f, indent=2)

    print(f"Output: {args.output}")
    print(f"Step: train={args.step}, val={args.val_step}")

    # Val sequences
    val_seq_set = set(train_seqs[::args.val_every])
    print(f"Val sequences: {len(val_seq_set)}")

    total_train = 0
    total_val = 0

    for seq in train_seqs:
        is_val = seq in val_seq_set
        print(f"\n{seq}" + (" [VAL]" if is_val else ""), flush=True)

        # Always write train
        n_train = process_sequence(seq, class_to_id, args, "train", train_img, train_lbl)
        total_train += n_train
        print(f"  train: {n_train} samples (total: {total_train})", flush=True)

        # Write val for selected sequences
        if is_val:
            n_val = process_sequence(seq, class_to_id, args, "val", val_img, val_lbl)
            total_val += n_val
            print(f"  val: {n_val} samples (total: {total_val})", flush=True)

    print(f"\nDone! train={total_train}, val={total_val}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
