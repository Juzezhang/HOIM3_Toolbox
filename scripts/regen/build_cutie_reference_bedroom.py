#!/usr/bin/env python3
"""Build Cutie reference masks for bedroom_data01 BAD views, using each view's
first-valid frame from mask_validity as the reference frame.

Inputs
------
- /simurgh2/datasets/HOI-M3/mask_shards/bedroom_data01/  (shard format; 7 objects)
- /simurgh/group/juze/datasets/HOI-M3/mask_validity/bedroom_data01/<frame:06d>.npz
    keys: <obj>_validity -> (42,) uint8

Outputs (per view)
------------------
- /simurgh2/datasets/HOI-M3/cutie_refs_bedroom_data01/bedroom_data01/masks/<v>.npy
    (1080, 1920) uint8 indexed (0=bg, 1..N = mask_shards object order)
- /simurgh2/datasets/HOI-M3/cutie_refs_bedroom_data01/bedroom_data01/masks/<v>_names.json
    {"mask_names": [...]}  (matches mask_shards meta.json objects order)
- /simurgh2/datasets/HOI-M3/cutie_refs_bedroom_data01/bedroom_data01/masks/<v>_start_frame.txt
    integer: the frame index ref was taken from (used as Cutie start_frame).
"""
from __future__ import annotations

import json
import os
import sys
from os.path import join

import numpy as np

# Local mask_io import
sys.path.insert(0, "/simurgh/u/juze/code/HOIM3_Toolbox")
from scripts.utils.mask_io import SequenceShardReaders, load_frame_masks_shard_full

SEQ = "bedroom_data01"
SHARD_ROOT = f"/simurgh2/datasets/HOI-M3/mask_shards/{SEQ}"
VALIDITY_DIR = f"/simurgh/group/juze/datasets/HOI-M3/mask_validity/{SEQ}"
OUT_ROOT = "/simurgh2/datasets/HOI-M3/cutie_refs_bedroom_data01"

# Bad views (union of person0<50% and person1<50%)
BAD_VIEWS = [0, 1, 2, 5, 6, 7, 8, 9, 23, 27, 28, 29, 30, 32, 33, 35, 37, 38, 40, 41]

TARGET_H = 1080
TARGET_W = 1920


def find_first_valid_frames():
    """For each bad view, find smallest frame id where person0_validity[V]==1
    and where person1_validity[V]==1.

    Returns dict: view -> {"p0": fid or None, "p1": fid or None, "ref": fid}
    """
    files = sorted(os.listdir(VALIDITY_DIR))
    files = [f for f in files if f.endswith(".npz")]
    p0 = {v: None for v in BAD_VIEWS}
    p1 = {v: None for v in BAD_VIEWS}
    search_p0 = set(BAD_VIEWS)
    search_p1 = set(BAD_VIEWS)
    for f in files:
        if not search_p0 and not search_p1:
            break
        fid = int(os.path.splitext(f)[0])
        z = np.load(join(VALIDITY_DIR, f))
        try:
            p0v = z["person0_validity"]
            p1v = z["person1_validity"]
        finally:
            z.close()
        done_p0 = [v for v in search_p0 if p0v[v] == 1]
        done_p1 = [v for v in search_p1 if p1v[v] == 1]
        for v in done_p0:
            p0[v] = fid
        for v in done_p1:
            p1[v] = fid
        search_p0 -= set(done_p0)
        search_p1 -= set(done_p1)

    info = {}
    for v in BAD_VIEWS:
        a, b = p0[v], p1[v]
        if a is None and b is None:
            info[v] = {"p0": None, "p1": None, "ref": None}
        elif a is None:
            info[v] = {"p0": None, "p1": b, "ref": b}
        elif b is None:
            info[v] = {"p0": a, "p1": None, "ref": a}
        else:
            info[v] = {"p0": a, "p1": b, "ref": max(a, b)}
    return info


def build_view_ref(seq_readers, view, ref_frame, out_dir):
    """Build (1080,1920) uint8 indexed mask for ONE view at ref_frame, and save
    .npy + _names.json + _start_frame.txt."""
    objects = list(seq_readers.objects)  # canonical order from meta.json
    # load_frame_masks_shard_full returns {obj: (V, H, W) uint8 {0,255}}
    full = load_frame_masks_shard_full(seq_readers, ref_frame)
    indexed = np.zeros((TARGET_H, TARGET_W), dtype=np.uint8)
    summary = []
    for cid, obj in enumerate(objects, start=1):
        m = full[obj][view]  # (H, W) uint8 {0,255}
        nz = int((m > 0).sum())
        if nz > 0:
            indexed[m > 0] = cid
            summary.append((cid, obj, nz))
    np.save(join(out_dir, f"{view}.npy"), indexed)
    with open(join(out_dir, f"{view}_names.json"), "w") as f:
        json.dump({"mask_names": objects}, f)
    with open(join(out_dir, f"{view}_start_frame.txt"), "w") as f:
        f.write(str(ref_frame) + "\n")
    return summary


def main():
    out_dir = join(OUT_ROOT, SEQ, "masks")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[ref] computing first-valid frames for {len(BAD_VIEWS)} bad views")
    info = find_first_valid_frames()
    print("View | p0_first | p1_first | ref")
    for v in BAD_VIEWS:
        e = info[v]
        print(f"  {v:>3} | {str(e['p0']):>8} | {str(e['p1']):>8} | {e['ref']}")
    skipped = [v for v in BAD_VIEWS if info[v]["ref"] is None]
    if skipped:
        print(f"SKIP views (no valid frame ever): {skipped}")

    sr = SequenceShardReaders(SHARD_ROOT)
    print(f"shard objects: {sr.objects}")
    print(f"shard meta: views={sr.views} h={sr.height} w={sr.width} num_frames={sr.meta['num_frames']}")

    summary_all = {}
    for v in BAD_VIEWS:
        rf = info[v]["ref"]
        if rf is None:
            continue
        summary = build_view_ref(sr, v, rf, out_dir)
        present = ",".join(f"{name}({cid}):{nz}" for cid, name, nz in summary)
        print(f"  v{v} ref_frame={rf} -> {present}")
        summary_all[v] = {"ref_frame": rf, "objects": summary}

    sr.close()

    with open(join(OUT_ROOT, SEQ, ".ref_built"), "w") as f:
        json.dump(
            {
                "seq": SEQ,
                "bad_views": BAD_VIEWS,
                "info": {str(k): {kk: vv for kk, vv in v.items()} for k, v in info.items()},
                "n_built": len(summary_all),
                "skipped": skipped,
                "objects": list(sr.objects) if False else None,
            },
            f,
            indent=2,
            default=int,
        )
    print(f"OK: {len(summary_all)} views built (skipped {len(skipped)})")


if __name__ == "__main__":
    main()
