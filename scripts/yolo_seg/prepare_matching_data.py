#!/usr/bin/env python3
"""Step 4: Build cross-view person matching dataset from mask shards.

For each training frame, selects K diverse camera views, arranges them in a
grid composite image, extracts person bboxes from masks, and maps them to
composite coordinates.  Output: WebDataset shards for matching network training.

Usage:
    python scripts/yolo_seg/prepare_matching_data.py --step 60 --num_views 6 --augment 3
"""

import argparse
import json
import math
import os
import random
import sys
from os.path import join

import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

import webdataset as wds  # noqa: E402

from scripts.utils.mask_io import SequenceShardReaders, load_frame_masks_shard_full  # noqa: E402
from scripts.visualize_mask_validity import preload_video_frames, resolve_view_validity  # noqa: E402
from scripts.yolo_seg.config import (  # noqa: E402
    SHARD_ROOT, VALIDITY_ROOT, VIDEO_ROOT, WDS_ROOT,
    MASK_H, MASK_W, NUM_VIEWS, FRAME_STEP, SAMPLES_PER_SHARD,
    load_sequence_contents, get_train_sequences,
)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Prepare cross-view person matching data")
    p.add_argument("--step", type=int, default=FRAME_STEP)
    p.add_argument("--num_views", type=int, default=6, help="Views per composite (K)")
    p.add_argument("--augment", type=int, default=3,
                   help="Random composites per frame (view-sampling augmentation)")
    p.add_argument("--view_w", type=int, default=320, help="Per-view width in composite")
    p.add_argument("--view_h", type=int, default=180, help="Per-view height in composite")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--sequences", nargs="+", default=None)
    p.add_argument("--min_person_views", type=int, default=2,
                   help="Min views a person must be visible in to be included")
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _grid_layout(K):
    cols = min(K, 3)
    rows = math.ceil(K / cols)
    return rows, cols


def _build_composite(view_frames, view_indices, vw, vh):
    """Arrange view frames into a row-major grid."""
    K = len(view_indices)
    rows, cols = _grid_layout(K)
    comp = np.zeros((rows * vh, cols * vw, 3), dtype=np.uint8)
    for i, v in enumerate(view_indices):
        if v not in view_frames:
            continue
        r, c = divmod(i, cols)
        frame = view_frames[v]
        if frame.shape[:2] != (vh, vw):
            frame = cv2.resize(frame, (vw, vh))
        comp[r * vh:(r + 1) * vh, c * vw:(c + 1) * vw] = frame
    return comp, rows, cols


def _mask_bbox(mask_binary):
    """Return (cx, cy, w, h) in mask-pixel coords, or None."""
    if not np.any(mask_binary):
        return None
    x, y, w, h = cv2.boundingRect(mask_binary.astype(np.uint8) * 255)
    if w == 0 or h == 0:
        return None
    return (x + w / 2.0, y + h / 2.0, float(w), float(h))


def _sample_diverse_views(valid_views, K):
    """Stratified sampling of K views spread across the camera ring."""
    if len(valid_views) <= K:
        return sorted(valid_views)
    bin_size = NUM_VIEWS / K
    selected = []
    for b in range(K):
        lo, hi = int(b * bin_size), int((b + 1) * bin_size)
        cands = [v for v in valid_views if lo <= v < hi]
        if cands:
            selected.append(random.choice(cands))
    remaining = [v for v in valid_views if v not in selected]
    while len(selected) < K and remaining:
        v = random.choice(remaining)
        selected.append(v)
        remaining.remove(v)
    return sorted(selected[:K])


# ── Per-sequence processing ───────────────────────────────────────────────────

def process_sequence(seq_name, args, shard_writer):
    shard_path = join(SHARD_ROOT, seq_name)
    validity_dir = join(VALIDITY_ROOT, seq_name)
    videos_dir = join(VIDEO_ROOT, seq_name, "videos")

    if not os.path.isdir(shard_path):
        return 0

    seq_readers = SequenceShardReaders(shard_path)
    frame_ids = seq_readers.frame_ids_list
    target_frames = frame_ids[:: args.step]
    if not target_frames:
        seq_readers.close()
        return 0

    person_objs = sorted(o for o in seq_readers.objects if o.startswith("person"))
    if not person_objs:
        seq_readers.close()
        return 0

    # Preload view frames
    tgt = (args.view_w, args.view_h)
    view_cache: dict[int, dict[int, np.ndarray]] = {}

    def _load(v):
        vp = join(videos_dir, f"{v}.mp4")
        return (v, preload_video_frames(vp, target_frames, tgt)) if os.path.exists(vp) else (v, {})

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for v, frames in pool.map(lambda v: _load(v), range(NUM_VIEWS)):
            if frames:
                view_cache[v] = frames

    all_views = list(range(NUM_VIEWS))
    rows, cols = _grid_layout(args.num_views)
    comp_w = cols * args.view_w
    comp_h = rows * args.view_h
    count = 0

    for frame_id in tqdm(target_frames, desc=f"  {seq_name}", leave=False):
        try:
            masks = load_frame_masks_shard_full(seq_readers, frame_id)
        except KeyError:
            continue

        # Load validity
        validity: dict[str, np.ndarray] = {}
        vfile = join(validity_dir, f"{frame_id:06d}.npz")
        if os.path.exists(vfile):
            with np.load(vfile) as vd:
                validity = {k: vd[k] for k in vd.files}

        # Per-person: which views are valid + bboxes
        person_views: dict[str, list[int]] = {}
        person_bboxes: dict[str, dict[int, tuple]] = {}
        for pname in person_objs:
            if pname not in masks:
                continue
            bbs: dict[int, tuple] = {}
            for v in all_views:
                vkey = f"{pname}_validity"
                if vkey in validity:
                    if not resolve_view_validity(validity[vkey], v, v, all_views):
                        continue
                bb = _mask_bbox(masks[pname][v] > 127)
                if bb is not None:
                    bbs[v] = bb
            if len(bbs) >= args.min_person_views:
                person_views[pname] = sorted(bbs.keys())
                person_bboxes[pname] = bbs

        if not person_views:
            continue

        # Union of all valid views
        union_views = sorted(set(v for vl in person_views.values() for v in vl))
        if len(union_views) < 2:
            continue

        # Generate augmented composites with different view subsets
        for aug_idx in range(args.augment):
            sel = _sample_diverse_views(union_views, args.num_views)

            # Check at least one person is visible in ≥2 selected views
            persons_ann = []
            for pname, bbs in person_bboxes.items():
                vis = [v for v in sel if v in bbs]
                if len(vis) < 2:
                    continue
                pid = int(pname.replace("person", ""))
                dets = []
                for v in vis:
                    cx, cy, w, h = bbs[v]
                    gp = sel.index(v)
                    r, c = divmod(gp, cols)
                    sx, sy = args.view_w / MASK_W, args.view_h / MASK_H
                    dets.append({
                        "view_idx": int(v),
                        "grid_pos": gp,
                        "bbox": [
                            (cx * sx + c * args.view_w) / comp_w,
                            (cy * sy + r * args.view_h) / comp_h,
                            (w * sx) / comp_w,
                            (h * sy) / comp_h,
                        ],
                    })
                persons_ann.append({"person_id": pid, "detections": dets})

            if not persons_ann:
                continue

            # Build composite image
            vf = {v: view_cache[v][frame_id] for v in sel
                  if v in view_cache and frame_id in view_cache[v]}
            composite, _, _ = _build_composite(vf, sel, args.view_w, args.view_h)

            key = f"{seq_name}_{frame_id:06d}_aug{aug_idx}"
            ok, jpeg = cv2.imencode(".jpg", composite, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                continue

            ann = json.dumps({
                "views": sel,
                "grid_rows": rows,
                "grid_cols": cols,
                "view_w": args.view_w,
                "view_h": args.view_h,
                "persons": persons_ann,
            }).encode("utf-8")

            shard_writer.write({"__key__": key, "jpg": jpeg.tobytes(), "json": ann})
            count += 1

    seq_readers.close()
    return count


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    seq_contents = load_sequence_contents()
    train_seqs = get_train_sequences(seq_contents)
    if args.sequences:
        allowed = set(args.sequences)
        train_seqs = [s for s in train_seqs if s in allowed]

    out_dir = join(WDS_ROOT, "matching_shards")
    os.makedirs(out_dir, exist_ok=True)

    writer = wds.ShardWriter(join(out_dir, "shard-%06d.tar"), maxcount=SAMPLES_PER_SHARD)
    total = 0
    for seq in train_seqs:
        print(f"\n{seq}", flush=True)
        n = process_sequence(seq, args, writer)
        total += n
        print(f"  -> {n} samples (cumulative {total})", flush=True)

    writer.close()
    print(f"\nDone. {total} matching samples in {out_dir}")


if __name__ == "__main__":
    main()
