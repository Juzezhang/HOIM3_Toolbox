#!/usr/bin/env python3
"""Restore mask_shards to pre-swap_fix state via inverse permutation (Path B).

For sequences with NO mask_npz backup, the only way to recover the original
person identity assignment is to invert the permutation that swap_fix's PASS2
applied. Since PASS2 only relabeled person0/1/.../personN (it did NOT modify
pixel content), an inverse permutation fully reverts.

Algorithm
---------
For each frame fid:
  perm = mapping[fid]           # from .swap_mapping.json
  inv  = argsort(perm)          # inverse permutation
  # After PASS2: cur[k] = orig[perm[k]]
  # Inverse:     orig[i] = cur[inv[i]]
  Read person_k shard frames from CURRENT shards.
  Write person_i := frame_data of person_{inv[i]}.

Implementation
--------------
1. Atomic rewrite: read full shard for each person object, write to
   {obj}.shard.new, then atomic os.replace.
2. Non-person objects are left untouched (shard files not modified).
3. .merged_1080p_done sentinel is preserved (still 1080p).
4. meta.json is preserved (objects list unchanged — only frame content rotated
   across person shards).

Usage
-----
    python restore_inverse_perm.py --seq office_data61 \
        --mapping_root /simurgh2/datasets/HOI-M3/mask_npz_generated \
        --shard_root   /simurgh2/datasets/HOI-M3/mask_shards
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from os.path import join

import numpy as np

sys.path.insert(0, "/simurgh/u/juze/code/HOIM3_Toolbox")
from scripts.utils.mask_io import (  # noqa: E402
    ShardReader,
    ShardWriter,
)


def restore_one_seq(
    seq: str,
    mapping_root: str,
    shard_root: str,
    compression_level: int = 6,
    dry_run: bool = False,
) -> str:
    seq_shard_dir = join(shard_root, seq)
    meta_path = join(seq_shard_dir, "meta.json")
    mapping_path = join(mapping_root, seq, ".swap_mapping.json")

    if not os.path.isfile(meta_path):
        return f"FAIL {seq}: no meta.json at {meta_path}"
    if not os.path.isfile(mapping_path):
        return f"FAIL {seq}: no .swap_mapping.json at {mapping_path}"

    with open(meta_path) as f:
        meta = json.load(f)
    with open(mapping_path) as f:
        mp = json.load(f)

    person_keys = mp["person_keys"]
    mappings = mp["mappings"]  # {"000000.npz": [perm], ...}
    objects = meta["objects"]
    views = meta["views"]
    H = meta["height"]
    W = meta["width"]
    frame_ids = meta["frame_ids"]

    # Validate: person_keys subset of objects
    for pk in person_keys:
        if pk not in objects:
            return f"FAIL {seq}: person key {pk} not in shard objects {objects}"

    # Build inverse perm per fid
    identity = list(range(len(person_keys)))
    inv_by_fid = {}
    nontriv = 0
    for fname, perm in mappings.items():
        # fname is "000000.npz"
        fid = int(os.path.splitext(fname)[0])
        if list(perm) == identity:
            inv_by_fid[fid] = identity
        else:
            inv_by_fid[fid] = list(np.argsort(perm).tolist())
            nontriv += 1

    # Frames missing from mapping → identity (safe default)
    missing = [fid for fid in frame_ids if fid not in inv_by_fid]
    if missing:
        for fid in missing:
            inv_by_fid[fid] = identity

    print(
        f"[{seq}] frames={len(frame_ids)} nontriv_inv={nontriv} "
        f"missing_in_mapping={len(missing)} persons={person_keys}",
        flush=True,
    )

    if dry_run:
        return f"DRY {seq}: nontriv={nontriv} (would rewrite person shards)"

    if nontriv == 0:
        return f"SKIP {seq}: all identity perms (nothing to restore)"

    # Open readers (current state) for person shards
    readers = {
        pk: ShardReader(join(seq_shard_dir, f"{pk}.shard")) for pk in person_keys
    }
    # Open writers to .shard.new for person shards
    writers = {
        pk: ShardWriter(
            join(seq_shard_dir, f"{pk}.shard.new"),
            len(frame_ids),
            compression_level=compression_level,
        )
        for pk in person_keys
    }
    for w in writers.values():
        w.__enter__()

    t0 = time.time()
    n = 0

    def _raw_compressed_bytes(reader: ShardReader, fid: int) -> bytes:
        """Read raw LZ4-compressed bytes for a frame (no decompress)."""
        offset, comp_size = reader.frame_index[fid]
        reader.f.seek(offset)
        return reader.f.read(comp_size)

    try:
        for fid in frame_ids:
            inv = inv_by_fid[fid]
            if inv == identity:
                # Identity: copy raw compressed bytes straight through (no decode)
                for pk in person_keys:
                    raw = _raw_compressed_bytes(readers[pk], fid)
                    writers[pk].write_frame_compressed(fid, raw)
            else:
                # Non-identity: still raw-bytes; just route src→dst by perm
                # restored.person_i := cur.person_{inv[i]}
                for i in range(len(person_keys)):
                    src_pk = person_keys[inv[i]]
                    dst_pk = person_keys[i]
                    raw = _raw_compressed_bytes(readers[src_pk], fid)
                    writers[dst_pk].write_frame_compressed(fid, raw)
            n += 1
            if n % 2000 == 0:
                dt = time.time() - t0
                rate = n / dt
                eta = (len(frame_ids) - n) / rate
                print(
                    f"  [{seq}] {n}/{len(frame_ids)} frames, "
                    f"{rate:.1f} fps, ETA {eta/60:.1f} min",
                    flush=True,
                )
    except Exception as e:
        for w in writers.values():
            try:
                w.__exit__(None, None, None)
            except Exception:
                pass
        for pk in person_keys:
            np_new = join(seq_shard_dir, f"{pk}.shard.new")
            if os.path.exists(np_new):
                os.remove(np_new)
        for r in readers.values():
            r.close()
        return f"FAIL {seq}: {type(e).__name__}: {e}"

    for w in writers.values():
        w.__exit__(None, None, None)
    for r in readers.values():
        r.close()

    # Atomic rename .shard.new → .shard for all person shards
    for pk in person_keys:
        old = join(seq_shard_dir, f"{pk}.shard")
        new = join(seq_shard_dir, f"{pk}.shard.new")
        os.replace(new, old)

    dt = time.time() - t0
    return f"OK {seq}: {len(frame_ids)} frames in {dt:.0f}s ({nontriv} nontriv inv)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument(
        "--mapping_root", default="/simurgh2/datasets/HOI-M3/mask_npz_generated"
    )
    ap.add_argument(
        "--shard_root", default="/simurgh2/datasets/HOI-M3/mask_shards"
    )
    ap.add_argument("--compression_level", type=int, default=6)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    result = restore_one_seq(
        args.seq,
        args.mapping_root,
        args.shard_root,
        compression_level=args.compression_level,
        dry_run=args.dry_run,
    )
    print(result, flush=True)


if __name__ == "__main__":
    main()
