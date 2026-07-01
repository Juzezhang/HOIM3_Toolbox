#!/usr/bin/env python3
"""Step 3 of Cutie mask-tracking pipeline: aggregate per-(seq,view) tracked
indexed masks into the canonical ``mask_npz`` format expected by
``convert_masks_npz_to_lz4.py``.

Input  (Step-2 output)
----------------------
    <cutie_root>/<seq>/<view>/<frame:06d>.npz
        keys: "mask" -> (1080, 1920) uint8 indexed (0=bg, 1..N=objects)
              "names" -> array of object names with names[0]="background", names[1..]=objects

Output (mirrors mask_npz_generated layout but at 1080p, BITPACKED)
-------------------------------------------------------
    <output_root>/<seq>/<frame:06d>.npz
        keys: <obj_name>          -> (n_views, 1080, W//8) uint8, big-endian bitpacked
              <obj_name>__shape   -> (3,) int64 = original (n_views, 1080, 1920)
              (background NOT included as a key)

Per-frame: read 42 views' indexed npz, split into per-object 0/255 masks,
stack across views, BITPACK along last axis (binary mask -> 1 bit/pixel),
save as a single npz with one key per object plus a __shape sibling key.

This is ~8x smaller than raw uint8 (1.2GB raw uint8 with 9 objects -> ~150MB),
making NFS writes proportionally faster.

Idempotent (skips frame if output exists). Convert script auto-detects the
bitpacked format via the __shape key.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from os.path import join

import numpy as np

# Per-frame inner-parallel reads: each frame reads 42 NFS .npz files. Doing them
# serially is ~1.3s of NFS RTT. With 8 threads we issue ~6 batches → ~200ms/frame.
INNER_READ_THREADS = 8

TARGET_H = 1080
TARGET_W = 1920


def _load_view_meta(cutie_root, seq):
    """Read .tracked_done for each view; return list of views and shared object names + min frames."""
    seq_dir = join(cutie_root, seq)
    view_dirs = sorted(
        [d for d in os.listdir(seq_dir) if d.isdigit() and os.path.isdir(join(seq_dir, d))],
        key=lambda x: int(x),
    )
    if not view_dirs:
        raise RuntimeError(f"No view dirs in {seq_dir}")
    metas = {}
    names_ref = None
    min_frames = None
    for v in view_dirs:
        sentinel = join(seq_dir, v, ".tracked_done")
        if not os.path.isfile(sentinel):
            raise RuntimeError(f"View {v} not done (no {sentinel})")
        with open(sentinel) as f:
            meta = json.load(f)
        metas[int(v)] = meta
        if names_ref is None:
            names_ref = list(meta["names"])
        else:
            if list(meta["names"]) != names_ref:
                raise RuntimeError(
                    f"View {v} names {meta['names']} differ from ref {names_ref}"
                )
        nf = int(meta["frames_saved"])
        min_frames = nf if min_frames is None else min(min_frames, nf)
    return [int(v) for v in view_dirs], names_ref, min_frames, metas


def _aggregate_frame(cutie_root, seq, output_root, frame_id, views, names):
    """Read all views for one frame, output a single npz."""
    out_path = join(output_root, seq, f"{frame_id:06d}.npz")
    if os.path.isfile(out_path):
        return f"SKIP frame {frame_id}"

    # Objects (skip background which is names[0])
    obj_names = names[1:]
    n_views = len(views)
    n_objs = len(obj_names)
    # Allocate (n_views, H, W) per object
    out = {n: np.zeros((n_views, TARGET_H, TARGET_W), dtype=np.uint8) for n in obj_names}

    def _read_one(v):
        npz_path = join(cutie_root, seq, str(v), f"{frame_id:06d}.npz")
        with np.load(npz_path, allow_pickle=False) as d:
            return d["mask"].copy()  # (1080, 1920) uint8

    with ThreadPoolExecutor(max_workers=INNER_READ_THREADS) as tp:
        idx_masks = list(tp.map(_read_one, views))

    for vi, idx_mask in enumerate(idx_masks):
        for cid, name in enumerate(obj_names, start=1):
            out[name][vi] = (idx_mask == cid).astype(np.uint8) * 255

    # Bitpack along width axis to shrink output ~8x.
    # (n_views, H, W) uint8(0/255) -> bool -> packbits along axis=-1
    # -> (n_views, H, ceil(W/8)) uint8. Convert script unpacks via __shape.
    packed = {}
    for name, arr in out.items():
        arr_bool = arr > 0  # (n_views, H, W) bool
        arr_packed = np.packbits(arr_bool, axis=-1)  # big-endian default
        packed[name] = arr_packed
        packed[f"{name}__shape"] = np.asarray(arr.shape, dtype=np.int64)

    tmp_base = join(output_root, seq, f"{frame_id:06d}.tmp")
    # Bitpacked uncompressed: ~100MB per frame (was 700MB raw). NFS write ~250ms.
    np.savez(tmp_base, **packed)
    os.replace(tmp_base + ".npz", out_path)
    return f"OK frame {frame_id}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument(
        "--cutie_root", default="/simurgh2/datasets/HOI-M3/cutie_tracking"
    )
    ap.add_argument(
        "--output_root", default="/simurgh2/datasets/HOI-M3/mask_npz_cutie"
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--max_frames", type=int, default=-1, help="Cap (debug). -1 = all."
    )
    args = ap.parse_args()

    seq = args.seq
    views, names, min_frames, metas = _load_view_meta(args.cutie_root, seq)
    print(
        f"[{seq}] views={len(views)} min_frames={min_frames} objects={names[1:]}",
        flush=True,
    )
    nf = min_frames if args.max_frames < 0 else min(min_frames, args.max_frames)
    os.makedirs(join(args.output_root, seq), exist_ok=True)

    frame_ids = list(range(nf))
    t0 = time.time()

    if args.workers <= 1:
        for fid in frame_ids:
            r = _aggregate_frame(args.cutie_root, seq, args.output_root, fid, views, names)
            if fid % 500 == 0:
                print(f"  {r} (elapsed {time.time()-t0:.0f}s)", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(
                    _aggregate_frame,
                    args.cutie_root,
                    seq,
                    args.output_root,
                    fid,
                    views,
                    names,
                ): fid
                for fid in frame_ids
            }
            done = 0
            for fut in as_completed(futures):
                done += 1
                if done % 500 == 0 or done == len(futures):
                    dt = time.time() - t0
                    rate = done / max(dt, 1e-6)
                    eta = (len(futures) - done) / max(rate, 1e-6)
                    print(
                        f"  [{seq}] aggregated {done}/{len(futures)} "
                        f"({rate:.1f} fps, ETA {eta/60:.1f}min)",
                        flush=True,
                    )

    # Sentinel
    with open(join(args.output_root, seq, ".aggregated"), "w") as f:
        json.dump(
            {"seq": seq, "n_frames": nf, "views": views, "names": names},
            f,
        )
    print(
        f"OK {seq}: aggregated {nf} frames in {time.time()-t0:.0f}s -> {args.output_root}/{seq}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
