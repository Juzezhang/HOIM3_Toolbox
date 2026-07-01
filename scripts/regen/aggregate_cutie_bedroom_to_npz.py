#!/usr/bin/env python3
"""Aggregate Cutie-tracked masks for bedroom_data01 BAD views into a staging
mask_npz directory, merging with the existing mask_shards for the GOOD views.

Inputs
------
- Cutie tracking outputs (20 BAD views):
    /scr/juze/datasets/HOI-M3/cutie_tracking_bedroom_data01/bedroom_data01/<v>/<frame:06d>.npz
    with key "mask" = (1080, 1920) uint8 indexed (0=bg, 1..7 = obj order).
    Note: bad views may have frames missing BEFORE their start_frame; those
    frames keep using the existing shard data.
- Existing mask_shards for the GOOD 22 views (and pre-start frames):
    /simurgh2/datasets/HOI-M3/mask_shards/bedroom_data01/

Output (one npz per frame, mirroring mask_npz_generated layout but at 1080p)
----------------------------------------------------------------------------
    /simurgh2/datasets/HOI-M3/mask_npz_cutie/bedroom_data01_fixed/<frame:06d>.npz
        keys: <obj_name> -> (n_views=42, 1080, 1920) uint8 {0, 255}

For each bad view V:
- If a Cutie output exists at frame F, REPLACE person0/person1 from Cutie;
  keep bed/book/cushion/smallsofa/television from existing shard (per spec).
- If no Cutie output at frame F (i.e. F < start_frame[V]), keep ALL objects
  from existing shard (validity for these frames is 0 anyway).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from os.path import join

import numpy as np

sys.path.insert(0, "/simurgh/u/juze/code/HOIM3_Toolbox")
from scripts.utils.mask_io import SequenceShardReaders, load_frame_masks_shard_full

SEQ = "bedroom_data01"
CUTIE_ROOT = "/scr/juze/datasets/HOI-M3/cutie_tracking_bedroom_data01"
SHARD_ROOT = f"/simurgh2/datasets/HOI-M3/mask_shards/{SEQ}"
OUTPUT_ROOT = "/simurgh2/datasets/HOI-M3/mask_npz_cutie/bedroom_data01_fixed"
BAD_VIEWS = [0, 1, 2, 5, 6, 7, 8, 9, 23, 27, 28, 29, 30, 32, 33, 35, 37, 38, 40, 41]
PERSON_OBJECTS = {"person0", "person1"}  # only these are replaced from Cutie

TARGET_H = 1080
TARGET_W = 1920


def _load_cutie_indexed(view, frame_id):
    """Return (H, W) uint8 indexed mask or None if missing."""
    p = join(CUTIE_ROOT, SEQ, str(view), f"{frame_id:06d}.npz")
    if not os.path.isfile(p):
        return None
    with np.load(p, allow_pickle=False) as d:
        return d["mask"].copy()


def _aggregate_frame(frame_id, objects, n_views, names_to_cid):
    """Build (42, H, W) per object for one frame.

    Loads existing shard frame for all objects + views as base, then for each
    bad view replaces person0/person1 with Cutie output if available.
    """
    out_path = join(OUTPUT_ROOT, f"{frame_id:06d}.npz")
    if os.path.isfile(out_path):
        return f"SKIP {frame_id}"

    # Open shard readers in this worker.
    sr = SequenceShardReaders(SHARD_ROOT)
    try:
        base = load_frame_masks_shard_full(sr, frame_id)  # {obj: (42,H,W) uint8 0/255}
    finally:
        sr.close()

    out = {obj: base[obj].copy() for obj in objects}

    for v in BAD_VIEWS:
        idx = _load_cutie_indexed(v, frame_id)
        if idx is None:
            # No Cutie output (frame < start_frame[V]). Keep base for ALL objects in this view.
            continue
        # Replace ONLY person0/person1 channels for this view.
        for obj in PERSON_OBJECTS:
            if obj not in names_to_cid:
                continue
            cid = names_to_cid[obj]
            mask_2d = ((idx == cid).astype(np.uint8) * 255)
            out[obj][v] = mask_2d
        # Note: bed/book/cushion/smallsofa/television in this view are kept from shard.

    tmp = join(OUTPUT_ROOT, f"{frame_id:06d}.tmp")
    np.savez_compressed(tmp, **out)
    os.replace(tmp + ".npz", out_path)
    return f"OK {frame_id}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_frames", type=int, default=-1)
    args = ap.parse_args()

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    sr = SequenceShardReaders(SHARD_ROOT)
    objects = list(sr.objects)
    n_views = sr.views
    num_frames = sr.meta["num_frames"]
    sr.close()
    # Map object name → cid (1-based, in shard order)
    # Note: the Cutie ref was built using EXACTLY this object order
    # (see build_cutie_reference_bedroom.py). So cid for person0/person1 matches.
    names_to_cid = {obj: i + 1 for i, obj in enumerate(objects)}
    print(
        f"[{SEQ}] objects={objects} (person0={names_to_cid.get('person0')}, "
        f"person1={names_to_cid.get('person1')})",
        flush=True,
    )

    # Verify the cutie ref names match shard order (they were built that way).
    ref_names_path = join(
        "/simurgh2/datasets/HOI-M3/cutie_refs_bedroom_data01",
        SEQ, "masks", "1_names.json",
    )
    with open(ref_names_path) as f:
        ref_names = json.load(f)["mask_names"]
    if ref_names != objects:
        print(
            f"FATAL: ref_names {ref_names} != shard objects {objects}",
            flush=True,
        )
        return 1

    nf = num_frames if args.max_frames < 0 else min(num_frames, args.max_frames)
    print(f"[{SEQ}] aggregating {nf} frames over {args.workers} workers", flush=True)
    frame_ids = list(range(nf))
    t0 = time.time()

    if args.workers <= 1:
        for fid in frame_ids:
            r = _aggregate_frame(fid, objects, n_views, names_to_cid)
            if fid % 500 == 0:
                print(f"  {r} (elapsed {time.time()-t0:.0f}s)", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {
                ex.submit(_aggregate_frame, fid, objects, n_views, names_to_cid): fid
                for fid in frame_ids
            }
            done = 0
            for fut in as_completed(futs):
                done += 1
                if done % 500 == 0 or done == len(futs):
                    dt = time.time() - t0
                    rate = done / max(dt, 1e-6)
                    eta = (len(futs) - done) / max(rate, 1e-6)
                    print(
                        f"  [{SEQ}] aggregated {done}/{len(futs)} "
                        f"({rate:.1f} fps, ETA {eta/60:.1f}min)",
                        flush=True,
                    )

    with open(join(OUTPUT_ROOT, ".aggregated"), "w") as f:
        json.dump(
            {
                "seq": SEQ,
                "n_frames": nf,
                "n_views": n_views,
                "objects": objects,
                "bad_views": BAD_VIEWS,
            },
            f,
            indent=2,
        )
    print(
        f"OK {SEQ}: aggregated {nf} frames in {time.time()-t0:.0f}s -> {OUTPUT_ROOT}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
