#!/usr/bin/env python3
"""Step 1 of Cutie mask-tracking pipeline for 11 broken HOI-M3 swap24 seqs.

For each (seq, view) pair, find the EARLIEST frame at which ALL persons are
visible simultaneously in that view (>=100 nz pixels per person). Use that frame
as the Cutie reference. The reference's swap_mapping permutation is also taken
from that frame (not necessarily frame 0). Output an indexed mask + names +
start_frame.txt per view.

This addresses the bug where frame 0 sometimes lacks one or more persons (e.g.
office_data24 v0 has person0+person2 but person1 nz=0). Cutie cannot track an
object that was not present in the initial reference, so tracking from frame 0
permanently loses any person absent at frame 0.

Inputs (per seq)
----------------
- /simurgh2/datasets/HOI-M3/mask_npz_generated/<seq>/<frame:06d>.npz
    dict of <obj_name>: (n_views, 720, 1280) uint8 (0/255)
- /simurgh2/datasets/HOI-M3/mask_npz_generated/<seq>/.swap_mapping.json
    {"person_keys": [...], "mappings": {"<frame>.npz": [perm]}, ...}

Outputs (per seq, per view)
---------------------------
- <out_root>/<seq>/masks/<view>.npy             (1080, 1920) uint8 indexed mask
- <out_root>/<seq>/masks/<view>_names.json      {"mask_names": [...]}
- <out_root>/<seq>/masks/<view>_start_frame.txt int (per-view reference frame)
- <out_root>/<seq>/.ref_built                   sentinel with details

Algorithm
---------
Per view V:
  1. Coarse scan: f = 0, 100, 200, ... until first f where all persons have
     >= MIN_NZ pixels at view V.
  2. Refine: scan f-99 .. f (or up to the first f-100 if f>=100), pick the
     EARLIEST frame in that window with all persons visible.
  3. Fallback: if no frame has all persons visible, pick the earliest frame
     with the MAXIMUM number of persons visible (and log WARN).

The reference mask itself is then extracted from
mask_npz_generated/<seq>/<ref_frame:06d>.npz with the inverse-perm of that
specific frame applied to the person channels.

Usage
-----
    python build_cutie_reference.py --seq office_data32
    python build_cutie_reference.py --seq ALL   # all 11 broken seqs
"""
from __future__ import annotations

import argparse
import json
import os
from os.path import join

import cv2
import numpy as np

BROKEN_SEQS = [
    "office_data24",
    "office_data28",
    "office_data32",
    "office_data33",
    "office_data34",
    "office_data35",
    "office_data36",
    "office_data40",
    "office_data41",
    "office_data42",
    "office_data55",
]

TARGET_H = 1080
TARGET_W = 1920
MIN_NZ = 100  # per-person threshold for "visible"
COARSE_STEP = 100  # coarse scan stride
SCAN_HARD_CAP = 2000  # do not scan more than this many frames per view (also caps backward buffer memory at ~12GB)


def inverse_perm(perm):
    """Return inverse permutation. perm[i]=j means dst[i]=src[j]; inv[j]=i."""
    return list(np.argsort(perm).tolist())


def _person_visible_at(npz_path, person_keys, view):
    """Return list[bool] indicating per-person visibility (>= MIN_NZ nz) at view.

    Returns None if file missing / unreadable.
    """
    if not os.path.isfile(npz_path):
        return None
    try:
        d = np.load(npz_path)
        return [int((d[pk][view] > 0).sum()) >= MIN_NZ for pk in person_keys]
    except Exception:
        return None


def find_ref_frame_for_view(seq_dir, person_keys, view, n_frames):
    """Return (ref_frame:int, status:str). status in {"all_visible","fallback"}.

    See module docstring for algorithm.
    """
    # Phase 1: coarse scan
    coarse_hit = None
    best_count = -1
    best_count_frame = 0
    f = 0
    scan_limit = min(n_frames, SCAN_HARD_CAP)
    while f < scan_limit:
        npz_path = join(seq_dir, f"{f:06d}.npz")
        vis = _person_visible_at(npz_path, person_keys, view)
        if vis is not None:
            c = sum(vis)
            if c > best_count:
                best_count = c
                best_count_frame = f
            if all(vis):
                coarse_hit = f
                break
        f += COARSE_STEP

    if coarse_hit is None:
        # No frame in scan range has all persons visible. Fallback to frame
        # with maximum persons visible (earliest).
        return best_count_frame, "fallback"

    # Phase 2: refine — earliest frame in [coarse_hit - COARSE_STEP + 1, coarse_hit]
    lo = max(0, coarse_hit - COARSE_STEP + 1)
    hi = coarse_hit
    earliest = coarse_hit
    for ff in range(lo, hi + 1):
        npz_path = join(seq_dir, f"{ff:06d}.npz")
        vis = _person_visible_at(npz_path, person_keys, view)
        if vis is not None and all(vis):
            earliest = ff
            break
    return earliest, "all_visible"


def _apply_perm_to_persons(data_dict, person_keys, perm):
    """Return dict[person_key] = unscrambled (H, W) for the chosen view.

    orig.person_i := cur.person_{inv[i]}
    """
    if list(perm) == list(range(len(person_keys))):
        return {pk: data_dict[pk] for pk in person_keys}
    inv = inverse_perm(perm)
    out = {}
    for i, pk in enumerate(person_keys):
        src_pk = person_keys[inv[i]]
        out[pk] = data_dict[src_pk]
    return out


def build_reference_for_seq(seq, mask_npz_root, out_root):
    seq_dir = join(mask_npz_root, seq)
    swap_path = join(seq_dir, ".swap_mapping.json")

    if not os.path.isdir(seq_dir):
        return f"FAIL {seq}: no {seq_dir}"
    if not os.path.isfile(swap_path):
        return f"FAIL {seq}: no {swap_path}"

    with open(swap_path) as f:
        mp = json.load(f)
    person_keys = mp["person_keys"]
    mappings = mp.get("mappings", {})

    # Number of frames
    npz_files = sorted([x for x in os.listdir(seq_dir) if x.endswith(".npz")])
    if not npz_files:
        return f"FAIL {seq}: no NPZ frames in {seq_dir}"
    n_frames = len(npz_files)

    # Probe frame 0 for keys / shape / n_views
    f0_path = join(seq_dir, "000000.npz")
    d0 = np.load(f0_path)
    keys = list(d0.files)  # canonical object ordering
    for pk in person_keys:
        if pk not in keys:
            return f"FAIL {seq}: person_key {pk} not in NPZ keys {keys}"
    sample = d0[keys[0]]
    n_views = sample.shape[0]
    src_h, src_w = sample.shape[1], sample.shape[2]

    out_dir = join(out_root, seq, "masks")
    os.makedirs(out_dir, exist_ok=True)

    # Cache NPZ data per frame we open (small per-call; persistent for reused
    # ref frames across views).
    npz_cache = {}

    def get_npz(frame_idx):
        if frame_idx not in npz_cache:
            npz_cache[frame_idx] = np.load(join(seq_dir, f"{frame_idx:06d}.npz"))
        return npz_cache[frame_idx]

    summary_lines = []
    per_view_info = []  # list of (view, ref_frame, status, present_ids)

    for v in range(n_views):
        ref_frame, status = find_ref_frame_for_view(seq_dir, person_keys, v, n_frames)
        # Pull data for ref_frame
        data = get_npz(ref_frame)
        perm = mappings.get(f"{ref_frame:06d}.npz", list(range(len(person_keys))))
        person_data = _apply_perm_to_persons(
            {pk: data[pk] for pk in person_keys}, person_keys, perm
        )

        # Build indexed mask (object order = keys order, IDs 1..N)
        idx = np.zeros((src_h, src_w), dtype=np.uint8)
        present_ids = []
        for cid, key in enumerate(keys, start=1):
            if key in person_keys:
                m = person_data[key][v]
            else:
                m = data[key][v]
            nz = int((m > 0).sum())
            if nz > 0:
                idx[m > 0] = cid
                present_ids.append((cid, key, nz))
        idx_up = cv2.resize(idx, (TARGET_W, TARGET_H), interpolation=cv2.INTER_NEAREST)

        np.save(join(out_dir, f"{v}.npy"), idx_up)
        with open(join(out_dir, f"{v}_names.json"), "w") as f:
            json.dump({"mask_names": keys}, f)
        with open(join(out_dir, f"{v}_start_frame.txt"), "w") as f:
            f.write(str(ref_frame))

        per_view_info.append((v, ref_frame, status, present_ids, perm))
        if status == "fallback":
            summary_lines.append(
                f"  WARN {seq}/v{v}: no frame in scan range has all persons "
                f"visible; fallback ref_frame={ref_frame} "
                f"(persons present: "
                f"{[k for _, k, _ in present_ids if k in person_keys]})"
            )

    # Build summary header
    person_cids = {pk: i + 1 for i, pk in enumerate(keys) if pk in person_keys}
    header = "  view | ref_frame | status | " + " | ".join(
        f"{pk}({person_cids[pk]})" for pk in person_keys
    )
    lines = [
        f"OK {seq}: {n_views} views, {len(keys)} objects, {n_frames} frames ({keys})",
        header,
    ]
    for v, ref_frame, status, present, perm in per_view_info:
        pmap = {kk: nz for _, kk, nz in present}
        perm_tag = "id" if list(perm) == list(range(len(person_keys))) else "perm"
        row = (
            f"   {v:>3}  | {ref_frame:>9} | {status[:8]:>8} ({perm_tag}) | "
            + " | ".join(f"{pmap.get(pk, 0):>7}" for pk in person_keys)
        )
        lines.append(row)
    lines.extend(summary_lines)

    # Sentinel
    with open(join(out_root, seq, ".ref_built"), "w") as f:
        f.write("\n".join(lines))
        f.write("\n")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True, help="seq name or 'ALL' for 11 broken")
    ap.add_argument(
        "--mask_npz_root", default="/simurgh2/datasets/HOI-M3/mask_npz_generated"
    )
    ap.add_argument(
        "--out_root", default="/simurgh2/datasets/HOI-M3/cutie_refs"
    )
    args = ap.parse_args()
    seqs = BROKEN_SEQS if args.seq == "ALL" else [args.seq]
    for s in seqs:
        print(build_reference_for_seq(s, args.mask_npz_root, args.out_root), flush=True)


if __name__ == "__main__":
    main()
