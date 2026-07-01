#!/usr/bin/env python3
"""Fast Hungarian person-id swap fix — Pass1+Pass2 with multiprocessing prefetch.

Pass 1 (compute mappings):
  - N worker processes load NPZ + compute per-view person centroids in parallel
  - Main process consumes results in order, runs Hungarian, stores per-frame mapping
  - This decouples NPZ decompression latency from sequential Hungarian step
Pass 2 (apply mappings):
  - For frames with non-identity perm, M worker processes rewrite the NPZ
    (selectively re-encoding only person arrays).

Mappings are persisted to <npz_dir>/.swap_mapping.json so we can resume / audit.
"""
# Cap BLAS thread count BEFORE numpy import so workers don't oversubscribe cores.
import os as _os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
import argparse
import io
import json
import os
import re
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

import numpy as np
from scipy.optimize import linear_sum_assignment

PIX_THRESH = 1000


def _per_view_centroids(arr):
    V, H, W = arr.shape
    onezero = (arr > 0).astype(np.uint8)
    row_counts = onezero.sum(axis=2, dtype=np.int64)
    col_counts = onezero.sum(axis=1, dtype=np.int64)
    npix = row_counts.sum(axis=1)
    valid = npix >= PIX_THRESH
    if not valid.any():
        return {}
    rows = np.arange(H, dtype=np.float64)
    cols = np.arange(W, dtype=np.float64)
    cy = (row_counts.astype(np.float64) @ rows) / np.maximum(npix, 1)
    cx = (col_counts.astype(np.float64) @ cols) / np.maximum(npix, 1)
    out = {}
    for v in np.where(valid)[0]:
        out[int(v)] = (float(cy[v]), float(cx[v]), int(npix[v]))
    return out


def _detect_person_keys_in_zip(npz_path):
    with zipfile.ZipFile(npz_path) as z:
        names = z.namelist()
    keys = []
    for n in names:
        m = re.match(r"^(person\d+)\.npy$", n)
        if m:
            keys.append(m.group(1))
    return sorted(keys, key=lambda x: int(x[len("person"):]))


def _worker_compute_centroids(args):
    """Pass 1 worker: load NPZ, return per-person per-view centroid dicts."""
    fi, path, person_keys = args
    try:
        with np.load(path) as d:
            arrays = [d[pk] for pk in person_keys]
        centroids = [_per_view_centroids(a) for a in arrays]
        return fi, path, centroids, None
    except Exception as e:
        return fi, path, None, f"{type(e).__name__}: {e}"


def _build_cost(canon_centroids, det_centroids, num_p):
    cost = np.full((num_p, num_p), 1e9, dtype=np.float64)
    for k in range(num_p):
        if not canon_centroids[k]:
            for j in range(num_p):
                cost[k, j] = 1e6 if det_centroids[j] else 5e6
            continue
        for j in range(num_p):
            dj = det_centroids[j]
            if not dj:
                cost[k, j] = 5e5
                continue
            shared = set(canon_centroids[k].keys()) & set(dj.keys())
            if not shared:
                cost[k, j] = 1e6
                continue
            d_total = 0.0
            for v in shared:
                c1 = canon_centroids[k][v]
                c2 = dj[v]
                d_total += ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5
            cost[k, j] = d_total / len(shared)
    return cost


def _update_canon(canon_centroids, det_centroids, mapping, ema=0.5):
    for k, j in enumerate(mapping):
        dj = det_centroids[j]
        for v, c in dj.items():
            if v in canon_centroids[k]:
                old = canon_centroids[k][v]
                canon_centroids[k][v] = (
                    ema * old[0] + (1 - ema) * c[0],
                    ema * old[1] + (1 - ema) * c[1],
                    c[2],
                )
            else:
                canon_centroids[k][v] = c


def _worker_rewrite(args):
    """Pass 2 worker: rewrite NPZ in place with permuted person arrays."""
    path, person_keys, mapping = args
    try:
        # Load only person arrays
        with np.load(path) as d:
            src_arrays = {pk: np.array(d[pk], copy=True) for pk in person_keys}
        # Determine new person arrays
        new_persons = {
            person_keys[k]: src_arrays[person_keys[mapping[k]]]
            for k in range(len(person_keys))
        }
        tmp = path + ".tmp"
        person_set = set(person_keys)
        with zipfile.ZipFile(path, "r") as zin:
            with zipfile.ZipFile(tmp, "w",
                                 compression=zipfile.ZIP_DEFLATED,
                                 compresslevel=6) as zout:
                for info in zin.infolist():
                    name = info.filename
                    m = re.match(r"^(person\d+)\.npy$", name)
                    if m and m.group(1) in person_set:
                        pk = m.group(1)
                        arr = new_persons[pk]
                        buf = io.BytesIO()
                        np.lib.format.write_array(buf, arr, allow_pickle=False)
                        zout.writestr(name, buf.getvalue(),
                                      compress_type=zipfile.ZIP_DEFLATED,
                                      compresslevel=6)
                    else:
                        raw = zin.read(name)
                        zout.writestr(info, raw)
        os.replace(tmp, path)
        return path, True, None
    except Exception as e:
        return path, False, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--report_every", type=int, default=2000)
    ap.add_argument("--num_persons", type=int, default=0)
    ap.add_argument("--pass1_workers", type=int, default=4)
    ap.add_argument("--pass2_workers", type=int, default=4)
    ap.add_argument("--mapping_file", default=None)
    args = ap.parse_args()

    npz_dir = args.npz_dir
    files = sorted([f for f in os.listdir(npz_dir) if f.endswith(".npz")])
    if not files:
        print(f"[fix] {npz_dir}: no NPZ"); sys.exit(1)
    if args.limit > 0:
        files = files[:args.limit]

    if args.num_persons > 0:
        person_keys = [f"person{i}" for i in range(args.num_persons)]
    else:
        person_keys = _detect_person_keys_in_zip(os.path.join(npz_dir, files[0]))
    num_persons = len(person_keys)
    if num_persons < 2:
        print(f"[fix] only {num_persons} persons — nothing to do"); sys.exit(0)

    mapping_file = args.mapping_file or os.path.join(npz_dir, ".swap_mapping.json")
    print(f"[fix] {npz_dir}: {len(files)} frames, {num_persons} persons {person_keys}")
    print(f"[fix] mapping_file: {mapping_file}")
    print(f"[fix] pass1_workers={args.pass1_workers} pass2_workers={args.pass2_workers}")

    # ---------- PASS 1: compute mappings ----------
    canon = {k: {} for k in range(num_persons)}
    mappings = {}
    nontrivial = 0
    perm_hist = {}
    t0 = time.time()

    ctx = get_context("forkserver")
    args_iter = [(fi, os.path.join(npz_dir, f), person_keys)
                 for fi, f in enumerate(files)]

    # Use imap to get results in order
    with ctx.Pool(processes=args.pass1_workers) as pool:
        for fi, path, centroids, err in pool.imap(
                _worker_compute_centroids, args_iter, chunksize=8):
            if err:
                print(f"[fix] PASS1 FAIL {os.path.basename(path)}: {err}")
                # store identity mapping so we don't rewrite this frame
                mappings[os.path.basename(path)] = list(range(num_persons))
                continue
            det = centroids  # list[pid] -> dict[view]->(cy,cx,npix)
            if fi == 0:
                mapping = list(range(num_persons))
                _update_canon(canon, det, mapping, ema=0.0)
            else:
                cost = _build_cost(canon, det, num_persons)
                row_ind, col_ind = linear_sum_assignment(cost)
                mapping = [0] * num_persons
                for r, c in zip(row_ind, col_ind):
                    mapping[r] = int(c)
                _update_canon(canon, det, mapping, ema=0.5)
            mappings[os.path.basename(path)] = mapping
            if mapping != list(range(num_persons)):
                nontrivial += 1
                perm_hist[tuple(mapping)] = perm_hist.get(tuple(mapping), 0) + 1
            if (fi + 1) % args.report_every == 0:
                dt = time.time() - t0
                rate = (fi + 1) / dt
                eta = (len(files) - fi - 1) / rate if rate > 0 else float('inf')
                print(f"[fix] PASS1 {fi+1}/{len(files)} nontrivial={nontrivial} "
                      f"rate={rate:.1f}f/s eta={eta:.0f}s", flush=True)

    pass1_time = time.time() - t0
    print(f"[fix] PASS1 DONE {pass1_time:.0f}s — non-identity perms: {nontrivial}")
    print(f"[fix] top permutations: {sorted(perm_hist.items(), key=lambda x:-x[1])[:8]}")

    # Save mapping
    with open(mapping_file, "w") as f:
        json.dump({"person_keys": person_keys, "mappings": mappings,
                   "stats": {"non_identity": nontrivial,
                             "perm_hist": {str(k): v for k, v in perm_hist.items()}}},
                  f)
    print(f"[fix] mapping saved")

    if args.dry_run:
        print(f"[fix] DRY RUN, skipping pass 2")
        return

    # ---------- PASS 2: rewrite NPZ files (only non-identity ones) ----------
    rewrite_args = []
    identity = list(range(num_persons))
    for fname, mapping in mappings.items():
        if mapping == identity:
            continue
        rewrite_args.append((os.path.join(npz_dir, fname), person_keys, mapping))

    print(f"[fix] PASS2 will rewrite {len(rewrite_args)} files")
    t1 = time.time()
    written = 0
    failed = 0
    if rewrite_args:
        with ctx.Pool(processes=args.pass2_workers) as pool:
            for i, (path, ok, err) in enumerate(pool.imap_unordered(
                    _worker_rewrite, rewrite_args, chunksize=4)):
                if ok:
                    written += 1
                else:
                    failed += 1
                    print(f"[fix] PASS2 FAIL {path}: {err}")
                if (i + 1) % 1000 == 0:
                    dt = time.time() - t1
                    rate = (i + 1) / dt
                    eta = (len(rewrite_args) - i - 1) / rate if rate > 0 else float('inf')
                    print(f"[fix] PASS2 {i+1}/{len(rewrite_args)} "
                          f"rate={rate:.1f}f/s eta={eta:.0f}s", flush=True)
    pass2_time = time.time() - t1

    total_time = time.time() - t0
    print(f"\n[fix] PASS2 DONE {pass2_time:.0f}s")
    print(f"[fix] frames rewritten: {written}, failed: {failed}")
    print(f"[fix] TOTAL {total_time:.0f}s")


if __name__ == "__main__":
    main()
