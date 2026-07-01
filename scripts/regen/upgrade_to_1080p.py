#!/usr/bin/env python3
"""Upgrade 720p mask shards to 1080p.

Reads named-key 720p shards and writes 1080p shards (atomic via .shard.new).
Updates meta.json and writes .merged_1080p_done sentinel.

For each seq in --seqs:
  - Skip if already has sentinel + 1080p meta
  - Read each frame from each .shard, upscale view masks to 1080×1920 NEAREST
  - Write to .shard.new
  - Atomic rename .shard.new → .shard
  - Rewrite meta.json with height=1080, width=1920
  - Touch .merged_1080p_done sentinel
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

sys.path.insert(0, '/simurgh/u/juze/code/HOIM3_Toolbox')
from scripts.utils.mask_io import ShardReader, ShardWriter, compress_mask_frame

SHARD_ROOT = '/simurgh2/datasets/HOI-M3/mask_shards'
TARGET_H = 1080
TARGET_W = 1920


def upscale_views(mask_views: np.ndarray) -> np.ndarray:
    """Upscale (views, H, W) uint8 {0,255} → (views, 1080, 1920) uint8 {0,255}."""
    V, H, W = mask_views.shape
    if H == TARGET_H and W == TARGET_W:
        return mask_views
    out = np.zeros((V, TARGET_H, TARGET_W), dtype=np.uint8)
    for v in range(V):
        out[v] = cv2.resize(mask_views[v], (TARGET_W, TARGET_H),
                            interpolation=cv2.INTER_NEAREST)
    return out


def upgrade_seq(seq: str, dry_run: bool = False) -> str:
    seq_dir = join(SHARD_ROOT, seq)
    meta_path = join(seq_dir, 'meta.json')
    sentinel = join(seq_dir, '.merged_1080p_done')

    if not os.path.exists(meta_path):
        return f'FAIL {seq}: no meta.json'

    with open(meta_path) as f:
        meta = json.load(f)

    src_h = meta['height']
    src_w = meta['width']
    objects = meta['objects']
    views = meta['views']
    frame_ids = meta['frame_ids']

    # Already done?
    if src_h == TARGET_H and src_w == TARGET_W and os.path.exists(sentinel):
        return f'SKIP {seq}: already 1080p + sentinel'

    # Need upgrade
    if src_h == 720 and src_w == 1280:
        pass
    elif src_h == TARGET_H and src_w == TARGET_W:
        # Already 1080p — verify NON-EMPTY before writing sentinel
        readers = {obj: ShardReader(join(seq_dir, f'{obj}.shard')) for obj in objects}
        try:
            sample_fids = frame_ids[::max(1, len(frame_ids)//5)][:5]
            total_pixels = 0
            for fid in sample_fids:
                for obj in objects:
                    m = readers[obj].read_frame(fid, views, src_h, src_w)
                    total_pixels += int(m.sum())
            if total_pixels == 0:
                return f'FAIL {seq}: 1080p but ALL EMPTY (needs full fix from mask_npz_generated)'
        finally:
            for r in readers.values():
                r.close()
        if not dry_run:
            open(sentinel, 'w').close()
        return f'SENTINEL_ONLY {seq}: already 1080p (verified non-empty), wrote sentinel'
    else:
        return f'FAIL {seq}: unexpected size {src_w}x{src_h}'

    print(f'[{seq}] upgrade {src_w}x{src_h} → {TARGET_W}x{TARGET_H}, '
          f'{len(frame_ids)} frames, {len(objects)} objects', flush=True)

    if dry_run:
        return f'DRY {seq}'

    # Open all readers
    readers = {}
    writers = {}
    for obj in objects:
        readers[obj] = ShardReader(join(seq_dir, f'{obj}.shard'))
        writers[obj] = ShardWriter(join(seq_dir, f'{obj}.shard.new'),
                                   len(frame_ids), compression_level=6)
        writers[obj].__enter__()

    from concurrent.futures import ThreadPoolExecutor

    def process_object_frame(obj, fid):
        mask_720 = readers[obj].read_frame(fid, views, src_h, src_w)
        mask_1080 = upscale_views(mask_720)
        comp = compress_mask_frame(mask_1080, compression_level=6)
        return obj, fid, comp

    t0 = time.time()
    n = 0
    n_workers = min(len(objects), 8)
    try:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for fid in frame_ids:
                futures = [pool.submit(process_object_frame, obj, fid) for obj in objects]
                for fut in futures:
                    obj, _, comp = fut.result()
                    writers[obj].write_frame_compressed(fid, comp)
                n += 1
                if n % 500 == 0:
                    dt = time.time() - t0
                    rate = n / dt
                    eta = (len(frame_ids) - n) / rate
                    print(f'  [{seq}] {n}/{len(frame_ids)} frames, {rate:.1f} fps, ETA {eta/60:.1f} min',
                          flush=True)
    except Exception as e:
        # Clean up partial .shard.new
        for w in writers.values():
            try:
                w.__exit__(None, None, None)
            except Exception:
                pass
        for obj in objects:
            new_path = join(seq_dir, f'{obj}.shard.new')
            if os.path.exists(new_path):
                os.remove(new_path)
        return f'FAIL {seq}: {type(e).__name__}: {e}'
    finally:
        for r in readers.values():
            r.close()

    # Close writers (writes index)
    for w in writers.values():
        w.__exit__(None, None, None)

    # Atomic rename .shard.new → .shard
    for obj in objects:
        old = join(seq_dir, f'{obj}.shard')
        new = join(seq_dir, f'{obj}.shard.new')
        os.replace(new, old)

    # Update meta.json
    meta['height'] = TARGET_H
    meta['width'] = TARGET_W
    tmp_meta = meta_path + '.tmp'
    with open(tmp_meta, 'w') as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp_meta, meta_path)

    # Write sentinel
    open(sentinel, 'w').close()

    dt = time.time() - t0
    return f'OK {seq}: {len(frame_ids)} frames in {dt:.0f}s'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seqs', nargs='+', required=True)
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()

    for seq in args.seqs:
        result = upgrade_seq(seq, dry_run=args.dry_run)
        print(result, flush=True)


if __name__ == '__main__':
    main()
