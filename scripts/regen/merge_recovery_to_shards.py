#!/usr/bin/env python3
"""
Merge SAM3+Cutie recovery output into existing mask_shards by REPLACING ONE
object's `.shard` file in-place.

Background
----------
For office_data32 (shredder) and office_data33 (radio), YOLO+ReID silently
produced all-zero masks for those objects, but the rest of the seq's masks are
fine. We re-tracked the missing object via SAM3 cross-view + Cutie temporal,
yielding 1080p per-frame indexed masks in:

    /simurgh2/datasets/HOI-M3/cutie_tracking_recovery_<obj>/<seq>/<view>/<frame:06d>.npz
        keys: "mask" -> (1080, 1920) uint8 indexed (0=bg, 1=<obj>)

The existing mask_shards/<seq>/ has 9 .shard files; only <obj>.shard is all-zero.
We rebuild *just* that one .shard and atomic-overwrite — no other object shards
or meta.json are touched.

Usage:
    python merge_recovery_to_shards.py --seq office_data32 --obj shredder
    python merge_recovery_to_shards.py --seq office_data33 --obj radio

Sentinel:
    Writes mask_shards/<seq>/.recovery_merged_<obj> with timestamp + summary.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from os.path import join

import numpy as np

# Import shard I/O from the existing utility
_HOIM3 = "/simurgh/u/juze/code/HOIM3_Toolbox"
if _HOIM3 not in sys.path:
    sys.path.insert(0, _HOIM3)
from scripts.utils.mask_io import ShardWriter, compress_mask_frame  # noqa: E402

INNER_READ_THREADS = 8  # parallel NFS reads per frame (42 views)


def _load_recovery_meta(recovery_seq_dir: str):
    """Discover available views + min frame count from .tracked_done sentinels."""
    view_dirs = sorted(
        [d for d in os.listdir(recovery_seq_dir)
         if d.isdigit() and os.path.isdir(join(recovery_seq_dir, d))],
        key=lambda x: int(x),
    )
    if not view_dirs:
        raise RuntimeError(f"No view dirs in {recovery_seq_dir}")
    metas = {}
    min_frames = None
    for v in view_dirs:
        sentinel = join(recovery_seq_dir, v, ".tracked_done")
        if not os.path.isfile(sentinel):
            raise RuntimeError(f"View {v} has no .tracked_done ({sentinel})")
        with open(sentinel) as f:
            meta = json.load(f)
        metas[int(v)] = meta
        nf = int(meta.get("frames_saved", 0))
        min_frames = nf if min_frames is None else min(min_frames, nf)
    return [int(v) for v in view_dirs], min_frames, metas


def _build_frame_mask(
    recovery_seq_dir: str,
    frame_id: int,
    views: list,
    n_views_target: int,
    H: int,
    W: int,
) -> np.ndarray:
    """Read 42 per-view indexed npz files for one frame; return (n_views_target, H, W) uint8 0/255."""
    out = np.zeros((n_views_target, H, W), dtype=np.uint8)

    def _read_one(v):
        p = join(recovery_seq_dir, str(v), f"{frame_id:06d}.npz")
        if not os.path.isfile(p):
            return v, None
        with np.load(p, allow_pickle=False) as d:
            return v, d["mask"].copy()  # (H, W) uint8 indexed (0 or 1)

    with ThreadPoolExecutor(max_workers=INNER_READ_THREADS) as tp:
        results = list(tp.map(_read_one, views))

    for v, m in results:
        if m is None:
            continue
        if m.shape != (H, W):
            raise RuntimeError(
                f"Frame {frame_id} view {v}: shape {m.shape} != target ({H},{W})"
            )
        out[v] = (m > 0).astype(np.uint8) * 255
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--obj", required=True, help="e.g. shredder, radio")
    ap.add_argument(
        "--recovery_root_template",
        default="/simurgh2/datasets/HOI-M3/cutie_tracking_recovery_{obj}",
        help="Format template; {obj} is filled with --obj.",
    )
    ap.add_argument(
        "--shards_root", default="/simurgh2/datasets/HOI-M3/mask_shards"
    )
    ap.add_argument("--compression_level", type=int, default=6)
    ap.add_argument("--workers", type=int, default=8, help="Frame-parallel workers")
    ap.add_argument(
        "--max_frames", type=int, default=-1, help="Cap (debug). -1 = all."
    )
    ap.add_argument(
        "--dry_run", action="store_true",
        help="Compute and report counts but don't overwrite the .shard file.",
    )
    args = ap.parse_args()

    seq = args.seq
    obj = args.obj
    recovery_seq_dir = args.recovery_root_template.format(obj=obj) + f"/{seq}"
    shards_seq_dir = join(args.shards_root, seq)
    shard_path = join(shards_seq_dir, f"{obj}.shard")
    tmp_shard_path = shard_path + ".new"
    sentinel_path = join(shards_seq_dir, f".recovery_merged_{obj}")

    # ----- Validate inputs -----
    if not os.path.isdir(recovery_seq_dir):
        print(f"ERROR: recovery dir missing: {recovery_seq_dir}", file=sys.stderr)
        return 2
    if not os.path.isdir(shards_seq_dir):
        print(f"ERROR: shards dir missing: {shards_seq_dir}", file=sys.stderr)
        return 2
    if not os.path.isfile(shard_path):
        print(f"ERROR: target {obj}.shard not found: {shard_path}", file=sys.stderr)
        return 2

    meta_path = join(shards_seq_dir, "meta.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    if obj not in meta["objects"]:
        print(f"ERROR: obj '{obj}' not in meta.json objects: {meta['objects']}",
              file=sys.stderr)
        return 2
    H = int(meta["height"])
    W = int(meta["width"])
    n_views_target = int(meta["views"])
    num_frames_meta = int(meta["num_frames"])
    frame_ids_meta = list(meta["frame_ids"])

    views, min_frames, _ = _load_recovery_meta(recovery_seq_dir)
    print(f"[{seq}/{obj}] recovery views={len(views)} min_frames={min_frames}",
          flush=True)
    print(f"[{seq}/{obj}] target shards: H={H} W={W} V={n_views_target} "
          f"num_frames={num_frames_meta}", flush=True)

    if min_frames < num_frames_meta:
        print(f"WARN: recovery min_frames ({min_frames}) < shards num_frames "
              f"({num_frames_meta}); missing-tail frames will get all-zero mask",
              flush=True)
    if set(views) != set(range(n_views_target)):
        missing = sorted(set(range(n_views_target)) - set(views))
        print(f"WARN: recovery missing views {missing}; those views will be "
              f"all-zero (this object simply not visible there)", flush=True)

    n_frames = num_frames_meta if args.max_frames < 0 else min(num_frames_meta, args.max_frames)

    if args.dry_run:
        print(f"[dry_run] would rebuild {shard_path} with {n_frames} frames", flush=True)
        return 0

    # ----- Build & write new shard -----
    t0 = time.time()
    print(f"[{seq}/{obj}] start rebuild → {tmp_shard_path}", flush=True)
    with ShardWriter(tmp_shard_path, num_frames=num_frames_meta,
                     compression_level=args.compression_level) as w:
        # Frame-parallel: workers compress; main thread writes sequentially to preserve index order
        def _process(fid: int):
            mask = _build_frame_mask(recovery_seq_dir, fid, views, n_views_target, H, W)
            comp = compress_mask_frame(mask, compression_level=args.compression_level)
            return fid, comp

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            # Submit in chunks to bound memory
            CHUNK = 64
            for chunk_start in range(0, n_frames, CHUNK):
                chunk_fids = frame_ids_meta[chunk_start:chunk_start + CHUNK]
                futures = [pool.submit(_process, fid) for fid in chunk_fids]
                results = [f.result() for f in futures]
                # Sort by frame id (preserve order)
                results.sort(key=lambda x: x[0])
                for fid, comp in results:
                    w.write_frame_compressed(fid, comp)

                done = chunk_start + len(chunk_fids)
                dt = time.time() - t0
                rate = done / max(dt, 1e-6)
                eta = (n_frames - done) / max(rate, 1e-6)
                print(f"  [{seq}/{obj}] {done}/{n_frames} ({rate:.1f} fps, "
                      f"ETA {eta/60:.1f}min)", flush=True)

        # If we capped at max_frames, write zero frames for the rest so index is complete
        zero_mask = np.zeros((n_views_target, H, W), dtype=np.uint8)
        zero_comp = compress_mask_frame(zero_mask, compression_level=args.compression_level)
        for fid in frame_ids_meta[n_frames:]:
            w.write_frame_compressed(fid, zero_comp)

    # ----- Atomic swap -----
    old_size = os.path.getsize(shard_path)
    new_size = os.path.getsize(tmp_shard_path)
    print(f"[{seq}/{obj}] old size={old_size/1e6:.1f}MB new size={new_size/1e6:.1f}MB",
          flush=True)
    backup_path = shard_path + ".old_zero"
    os.replace(shard_path, backup_path)
    os.replace(tmp_shard_path, shard_path)
    print(f"[{seq}/{obj}] swapped: {shard_path} (kept old at {backup_path})", flush=True)

    # ----- Sentinel -----
    with open(sentinel_path, "w") as f:
        json.dump({
            "seq": seq,
            "obj": obj,
            "merged_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "recovery_views": views,
            "recovery_min_frames": min_frames,
            "shards_num_frames": num_frames_meta,
            "old_shard_size_bytes": old_size,
            "new_shard_size_bytes": new_size,
            "elapsed_seconds": time.time() - t0,
        }, f, indent=2)
    print(f"[{seq}/{obj}] DONE in {time.time()-t0:.0f}s → sentinel: {sentinel_path}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
