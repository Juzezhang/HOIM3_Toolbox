#!/usr/bin/env python3
"""Prepare cross-view person ReID dataset from mask shards.

For each frame, crops person regions from multiple views using mask bboxes.
Person IDs are consistent across views (from mask shard naming).

Output: standard image classification folder structure for ReID training:
    {output}/train/{seq}_{person_id}/view{v}_frame{f}.jpg
    {output}/val/...

Usage:
    python scripts/yolo_seg/prepare_reid_data.py --step 300 --workers 8
"""

import argparse
import json
import os
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
    SHARD_ROOT, VALIDITY_ROOT, VIDEO_ROOT, NUM_VIEWS,
    load_sequence_contents, get_train_sequences,
)


def parse_args():
    p = argparse.ArgumentParser(description="Prepare cross-view person ReID data")
    p.add_argument("--step", type=int, default=300, help="Frame step")
    p.add_argument("--crop_size", type=int, default=256, help="Crop resize (height)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--views", type=int, nargs="+", default=None,
                   help="Subset of views (default: all 42)")
    p.add_argument("--val_every", type=int, default=5)
    p.add_argument("--output", type=str, default="/simurgh2/datasets/HOI-M3/reid_data")
    p.add_argument("--sequences", nargs="+", default=None)
    return p.parse_args()


def crop_person(frame, mask_view, padding_ratio=0.1):
    """Crop person region from frame using mask bbox, with padding."""
    binary = mask_view > 127
    if not np.any(binary):
        return None
    u8 = binary.astype(np.uint8) * 255
    x, y, w, h = cv2.boundingRect(u8)
    if w < 10 or h < 10:
        return None

    # Add padding
    img_h, img_w = frame.shape[:2]
    mask_h, mask_w = mask_view.shape
    # Scale bbox from mask resolution to frame resolution
    sx, sy = img_w / mask_w, img_h / mask_h
    x1 = int(max(x * sx - w * sx * padding_ratio, 0))
    y1 = int(max(y * sy - h * sy * padding_ratio, 0))
    x2 = int(min((x + w) * sx + w * sx * padding_ratio, img_w))
    y2 = int(min((y + h) * sy + h * sy * padding_ratio, img_h))

    if x2 - x1 < 10 or y2 - y1 < 10:
        return None

    return frame[y1:y2, x1:x2]


def process_sequence(seq_name, args, split):
    """Process one sequence, output person crops."""
    shard_path = join(SHARD_ROOT, seq_name)
    validity_dir = join(VALIDITY_ROOT, seq_name)
    videos_dir = join(VIDEO_ROOT, seq_name, "videos")

    meta_file = join(shard_path, "meta.json")
    if not os.path.isfile(meta_file):
        return 0

    seq_readers = SequenceShardReaders(shard_path)
    frame_ids = seq_readers.frame_ids_list
    target_frames = frame_ids[::args.step]
    if not target_frames:
        seq_readers.close()
        return 0

    # Find person objects
    persons = sorted(o for o in seq_readers.objects if o.startswith("person"))
    if not persons:
        seq_readers.close()
        return 0

    view_list = args.views if args.views else list(range(NUM_VIEWS))
    tgt_size = (640, 360)  # video frame size

    # Preload video frames
    view_frames = {}
    def _load(v):
        vpath = join(videos_dir, f"{v}.mp4")
        if not os.path.exists(vpath):
            return v, {}
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            return v, {}
        results = {}
        for idx in sorted(target_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                results[idx] = cv2.resize(frame, tgt_size)
        cap.release()
        return v, results

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for v, frames in pool.map(lambda v: _load(v), view_list):
            if frames:
                view_frames[v] = frames

    all_views = list(range(NUM_VIEWS))
    crop_count = 0

    for frame_id in tqdm(target_frames, desc=f"  {seq_name}", leave=False):
        try:
            masks = load_frame_masks_shard_full(seq_readers, frame_id)
        except KeyError:
            continue

        # Load validity
        validity = {}
        vfile = join(validity_dir, f"{frame_id:06d}.npz")
        if os.path.exists(vfile):
            with np.load(vfile) as vdata:
                validity = {k: vdata[k] for k in vdata.files}

        for person_name in persons:
            if person_name not in masks:
                continue
            # Global person ID: seq_name + person_name
            pid = f"{seq_name}_{person_name}"
            pid_dir = join(args.output, split, pid)
            os.makedirs(pid_dir, exist_ok=True)

            for view_idx in view_list:
                if view_idx not in view_frames:
                    continue
                frame = view_frames[view_idx].get(frame_id)
                if frame is None:
                    continue

                # Check validity
                vkey = f"{person_name}_validity"
                if vkey in validity:
                    if not resolve_view_validity(validity[vkey], view_idx, view_idx, all_views):
                        continue

                # Crop person
                mask_view = masks[person_name][view_idx]
                crop = crop_person(frame, mask_view)
                if crop is None:
                    continue

                # Resize to standard height, keep aspect ratio
                h, w = crop.shape[:2]
                new_h = args.crop_size
                new_w = int(w * new_h / h)
                new_w = max(new_w, 64)  # minimum width
                crop = cv2.resize(crop, (new_w, new_h))

                # Save
                fname = f"v{view_idx:02d}_f{frame_id:06d}.jpg"
                cv2.imwrite(join(pid_dir, fname), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                crop_count += 1

    seq_readers.close()
    return crop_count


def main():
    args = parse_args()

    seq_contents = load_sequence_contents()
    train_seqs = get_train_sequences(seq_contents)
    if args.sequences:
        allowed = set(args.sequences)
        train_seqs = [s for s in train_seqs if s in allowed]

    val_seqs = set(train_seqs[::args.val_every])
    print(f"Sequences: {len(train_seqs)} ({len(val_seqs)} val)")
    print(f"Step: {args.step}, Views: {len(args.views) if args.views else 42}")
    print(f"Output: {args.output}")

    os.makedirs(join(args.output, "train"), exist_ok=True)
    os.makedirs(join(args.output, "val"), exist_ok=True)

    total_train = 0
    total_val = 0

    for seq in train_seqs:
        is_val = seq in val_seqs
        split = "val" if is_val else "train"
        n = process_sequence(seq, args, split)
        if is_val:
            total_val += n
        else:
            total_train += n
        print(f"  {seq} [{split}]: {n} crops (train={total_train}, val={total_val})", flush=True)

    # Write metadata
    meta = {
        "num_train_crops": total_train,
        "num_val_crops": total_val,
        "num_train_ids": len(os.listdir(join(args.output, "train"))),
        "num_val_ids": len(os.listdir(join(args.output, "val"))),
        "step": args.step,
        "crop_size": args.crop_size,
    }
    with open(join(args.output, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone!")
    print(f"Train: {total_train} crops, {meta['num_train_ids']} identities")
    print(f"Val: {total_val} crops, {meta['num_val_ids']} identities")


if __name__ == "__main__":
    main()
