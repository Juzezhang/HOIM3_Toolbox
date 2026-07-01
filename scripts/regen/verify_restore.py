#!/usr/bin/env python3
"""Verify restore_swap24 outputs.

For Path A: compare shard contents against /simurgh/group/mask_npz source
(downsampled to 1080p reference and back to NEAREST 1080p for fair comparison —
actually mask_npz is 1080p natively).

For Path B: compare against pre-restore shard with inverse perm applied
(round-trip identity).

Usage:
    python verify_restore.py --seq bedroom_data35 --path A
    python verify_restore.py --seq office_data61   --path B
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from os.path import join

import numpy as np

sys.path.insert(0, "/simurgh/u/juze/code/HOIM3_Toolbox")
from scripts.utils.mask_io import ShardReader  # noqa: E402

SHARD_ROOT = "/simurgh2/datasets/HOI-M3/mask_shards"
GROUP_NPZ_ROOT = "/simurgh/group/juze/datasets/HOI-M3/mask_npz"


def verify_path_a(seq: str, num_samples: int = 3) -> bool:
    seq_dir = join(SHARD_ROOT, seq)
    npz_dir = join(GROUP_NPZ_ROOT, seq)
    meta = json.load(open(join(seq_dir, "meta.json")))
    objects = meta["objects"]
    V, H, W = meta["views"], meta["height"], meta["width"]
    frame_ids = meta["frame_ids"]
    sentinel = join(seq_dir, ".merged_1080p_done")
    assert os.path.exists(sentinel), f"missing sentinel {sentinel}"
    print(
        f"[{seq}] meta: V={V} H={H} W={W} frames={len(frame_ids)} objs={len(objects)}"
    )
    assert H == 1080 and W == 1920, f"expected 1080p, got {H}x{W}"

    random.seed(42)
    sample = random.sample(frame_ids, min(num_samples, len(frame_ids)))
    readers = {obj: ShardReader(join(seq_dir, f"{obj}.shard")) for obj in objects}
    ok = True
    for fid in sample:
        npz_path = join(npz_dir, f"{fid:06d}.npz")
        if not os.path.isfile(npz_path):
            print(f"  fid={fid}: NPZ source missing {npz_path}")
            ok = False
            continue
        data = np.load(npz_path)
        for obj in objects:
            if obj not in data:
                print(f"  fid={fid} {obj}: missing in NPZ")
                ok = False
                continue
            src = data[obj]  # (V, H, W) 1080p
            shard_mask = readers[obj].read_frame(fid, V, H, W)
            # Convert both to binary
            src_bin = (src > 0).astype(np.uint8)
            shard_bin = (shard_mask > 0).astype(np.uint8)
            if not np.array_equal(src_bin, shard_bin):
                # mask_npz is 1080p; shard is NEAREST upsampled from 720p
                # so they won't be pixel-identical. Check IoU instead.
                inter = int((src_bin & shard_bin).sum())
                union = int((src_bin | shard_bin).sum())
                iou = inter / max(union, 1)
                if iou < 0.95:
                    print(f"  fid={fid} {obj}: IoU={iou:.3f} (LOW)")
                    ok = False
                else:
                    print(f"  fid={fid} {obj}: IoU={iou:.3f} OK (nearest-upsample artifact)")
            else:
                print(f"  fid={fid} {obj}: EXACT match")
        data.close()
    for r in readers.values():
        r.close()
    return ok


def verify_path_b(seq: str, num_samples: int = 3) -> bool:
    """Path B: assert non-empty restored shards + spot check that perm was applied.

    The pre-restore is in .swap_fix.bak. For non-identity frames, the restored
    person_i shard should equal the bak's person_{inv[i]} shard.
    """
    seq_dir = join(SHARD_ROOT, seq)
    bak_dir = join(SHARD_ROOT, f"{seq}.swap_fix.bak")
    meta = json.load(open(join(seq_dir, "meta.json")))
    objects = meta["objects"]
    V, H, W = meta["views"], meta["height"], meta["width"]
    frame_ids = meta["frame_ids"]
    print(
        f"[{seq}] meta: V={V} H={H} W={W} frames={len(frame_ids)} objs={len(objects)}"
    )

    mapping = json.load(
        open(
            f"/simurgh2/datasets/HOI-M3/mask_npz_generated/{seq}/.swap_mapping.json"
        )
    )
    person_keys = mapping["person_keys"]
    perms = mapping["mappings"]

    # Sample non-identity frames
    identity = list(range(len(person_keys)))
    nontriv_fids = [
        int(os.path.splitext(fname)[0])
        for fname, perm in perms.items()
        if list(perm) != identity
    ]
    if not nontriv_fids:
        print(f"  [{seq}] all identity perms — trivial restore")
        return True

    random.seed(42)
    sample = random.sample(nontriv_fids, min(num_samples, len(nontriv_fids)))

    cur_readers = {pk: ShardReader(join(seq_dir, f"{pk}.shard")) for pk in person_keys}
    bak_readers = {pk: ShardReader(join(bak_dir, f"{pk}.shard")) for pk in person_keys}

    ok = True
    for fid in sample:
        perm = perms[f"{fid:06d}.npz"]
        inv = list(np.argsort(perm).tolist())
        print(f"  fid={fid} perm={perm} inv={inv}")
        for i in range(len(person_keys)):
            # restored person_i should equal bak person_{inv[i]}
            cur = cur_readers[person_keys[i]].read_frame(fid, V, H, W)
            ref = bak_readers[person_keys[inv[i]]].read_frame(fid, V, H, W)
            same = np.array_equal(cur, ref)
            print(
                f"    person{i}: cur == bak.person{inv[i]} → {same} "
                f"(cur_nnz={(cur>0).sum()}, ref_nnz={(ref>0).sum()})"
            )
            if not same:
                ok = False

    for r in cur_readers.values():
        r.close()
    for r in bak_readers.values():
        r.close()
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--path", choices=["A", "B"], required=True)
    ap.add_argument("--num_samples", type=int, default=3)
    args = ap.parse_args()
    if args.path == "A":
        ok = verify_path_a(args.seq, args.num_samples)
    else:
        ok = verify_path_b(args.seq, args.num_samples)
    print(f"\n[{args.seq}] PATH {args.path} verify: {'OK' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
