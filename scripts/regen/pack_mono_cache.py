"""Pack per-frame HOI-M3 mono MHR NPZs into a packed cache for fast bulk-load.

Reads:  <mono_root>/<seq>/<view>/<person>/<frame:06d>.npz
Writes: <output_root>/<seq>/<person>/{keypoints2d_308.npy,
                                       keypoints2d_70.npy,
                                       shape_params.npy,
                                       model_parameters.npy,
                                       meta.npz}

Schema (matches load_keypoints2d_npz_hoim3_packed.py):
  keypoints2d_308 : (n_frames, n_views, 308, 3) float32
  keypoints2d_70  : (n_frames, n_views,  70, 3) float32
  shape_params    : (n_frames, n_views,      45) float32
  model_parameters: (n_frames, n_views,     204) float32
  meta.npz        : views=int array of view ids, frame_indices=arange(n_frames)

Idempotent: a .done sentinel + presence of all .npy files = skip.
"""
import argparse
import os
import sys
import time
import multiprocessing as mp
from functools import partial
from typing import Optional

import numpy as np

# Make sure mvbodyfit is on path.
sys.path.insert(0, "/simurgh/u/juze/code/mv-bodyfit")
from mvbodyfit.operations.mhr_sam3d import (
    build_mhr_model_parameters_from_sam3d_npz,
    load_sam3d_mhr_assets,
)


SAM3D_CKPT_DEFAULT = (
    "/simurgh/u/juze/code/fast-sam-3d-body/checkpoints/sam-3d-body-dinov3/model.ckpt"
)


def _load_one(args, assets):
    """Worker: load one npz, return (frame_idx, view_idx, person, k308, k70, shape, mp)
    or None on failure."""
    path, frame_idx, view_idx, person = args
    try:
        data = np.load(path, allow_pickle=True)
        # k308 (no per-file 308 keypoints in current mono output; stays zero)
        k308 = np.zeros((308, 3), dtype=np.float32)
        for src in ("pred_keypoints_2d_308", "keypoints2d_308"):
            if src in data.files:
                k = data[src]
                if k.shape == (308, 2):
                    k308[:, :2] = k
                    k308[:, 2] = 1.0
                elif k.shape == (308, 3):
                    k308 = k.astype(np.float32)
                break

        # k70
        k70 = np.zeros((70, 3), dtype=np.float32)
        used_k70 = False
        for src in ("pred_keypoints_2d", "keypoints2d_70"):
            if src in data.files:
                k = data[src]
                if k.shape == (70, 2):
                    k70[:, :2] = k
                    k70[:, 2] = 1.0
                elif k.shape == (70, 3):
                    k70 = k.astype(np.float32)
                used_k70 = True
                break
        if not used_k70:
            k70 = k308[:70].copy()

        # shape (45,)
        if "shape_params" in data.files:
            s = data["shape_params"].flatten()[:45].astype(np.float32)
            shape = np.zeros(45, dtype=np.float32)
            shape[: len(s)] = s
        else:
            shape = np.zeros(45, dtype=np.float32)

        # model_parameters (204,)
        mp_arr = np.zeros(204, dtype=np.float32)
        if assets is not None and (
            "body_pose_params" in data.files
            or "hand_pose_params" in data.files
            or "scale_params" in data.files
        ):
            try:
                mp_arr = build_mhr_model_parameters_from_sam3d_npz(
                    data, assets=assets, use_mono_global=False
                ).astype(np.float32)
            except Exception:
                mp_arr = np.zeros(204, dtype=np.float32)
        elif "model_parameters" in data.files:
            mpraw = np.asarray(data["model_parameters"], dtype=np.float32).reshape(-1)
            if mpraw.size >= 204:
                mp_arr = mpraw[:204]
            elif mpraw.size > 0:
                mp_arr[: mpraw.size] = mpraw

        return frame_idx, view_idx, person, k308, k70, shape, mp_arr
    except Exception as e:
        return None


# Worker globals (initialised once per process).
_W_ASSETS = None


def _init_worker(sam3d_ckpt: str):
    global _W_ASSETS
    if sam3d_ckpt and os.path.exists(sam3d_ckpt):
        try:
            _W_ASSETS = load_sam3d_mhr_assets(sam3d_ckpt)
        except Exception as e:
            print(f"[pack_mono] worker FAILED to load assets: {e}", flush=True)
            _W_ASSETS = None
    else:
        _W_ASSETS = None


def _worker_call(args):
    return _load_one(args, _W_ASSETS)


def _scan_frames(seq_mono_root: str, views: list, persons: list) -> int:
    """Determine n_frames by walking every view/person dir."""
    max_fi = -1
    for v in views:
        v_dir = os.path.join(seq_mono_root, str(v))
        if not os.path.isdir(v_dir):
            continue
        for person in persons:
            p_dir = os.path.join(v_dir, person)
            if not os.path.isdir(p_dir):
                continue
            for fn in os.listdir(p_dir):
                if not fn.endswith(".npz") or fn.endswith(".tmp.npz"):
                    continue
                try:
                    fi = int(fn[:-4])
                    if fi > max_fi:
                        max_fi = fi
                except ValueError:
                    continue
    return max_fi + 1 if max_fi >= 0 else 0


def _discover_persons(seq_mono_root: str, views: list) -> list:
    persons = set()
    for v in views:
        v_dir = os.path.join(seq_mono_root, str(v))
        if not os.path.isdir(v_dir):
            continue
        for entry in os.listdir(v_dir):
            if entry.startswith("person") and os.path.isdir(os.path.join(v_dir, entry)):
                persons.add(entry)
    return sorted(persons, key=lambda x: int(x.replace("person", "")))


def _is_complete(seq_out_dir: str, persons: list, mono_seq_dir: str = None) -> bool:
    if not persons:
        return False
    done_marker = os.path.join(seq_out_dir, ".pack_done")
    if not os.path.exists(done_marker):
        return False
    for person in persons:
        p_dir = os.path.join(seq_out_dir, person)
        # k308.npy is optional (skipped when all zero — current mono is 70-kp only).
        for name in (
            "keypoints2d_70.npy",
            "shape_params.npy",
            "model_parameters.npy",
            "meta.npz",
        ):
            if not os.path.exists(os.path.join(p_dir, name)):
                return False
    # Staleness check: if mono dir has any file newer than .pack_done, treat as stale.
    if mono_seq_dir is not None and os.path.isdir(mono_seq_dir):
        try:
            pack_mtime = os.path.getmtime(done_marker)
        except Exception:
            return False
        for v_name in sorted(os.listdir(mono_seq_dir)):
            if not v_name.isdigit():
                continue
            v_dir = os.path.join(mono_seq_dir, v_name)
            if not os.path.isdir(v_dir):
                continue
            for p_name in sorted(os.listdir(v_dir))[:1]:
                p_dir = os.path.join(v_dir, p_name)
                if not os.path.isdir(p_dir):
                    continue
                for fn in sorted(os.listdir(p_dir))[:1]:
                    try:
                        mt = os.path.getmtime(os.path.join(p_dir, fn))
                        if mt > pack_mtime + 60:  # mono > 60s newer
                            print(
                                f"[pack_mono] STALE: mono {v_name}/{p_name}/{fn} "
                                f"is {(mt - pack_mtime)/3600:.1f}h newer than .pack_done — re-pack",
                                flush=True,
                            )
                            return False
                    except Exception:
                        pass
                    break
                break
            break
    return True


def pack_sequence(
    sequence: str,
    mono_root: str,
    output_root: str,
    views: list,
    workers: int,
    sam3d_ckpt: str,
) -> bool:
    seq_mono_root = os.path.join(mono_root, sequence)
    if not os.path.isdir(seq_mono_root):
        print(f"[pack_mono] {sequence}: mono dir missing: {seq_mono_root}", flush=True)
        return False

    seq_out_dir = os.path.join(output_root, sequence)

    # Discover available persons (some views may be missing entirely).
    available_views = [v for v in views if os.path.isdir(os.path.join(seq_mono_root, str(v)))]
    if not available_views:
        print(f"[pack_mono] {sequence}: no requested views available", flush=True)
        return False

    persons = _discover_persons(seq_mono_root, available_views)
    if not persons:
        print(f"[pack_mono] {sequence}: no persons found", flush=True)
        return False

    if _is_complete(seq_out_dir, persons, mono_seq_dir=seq_mono_root):
        print(f"[pack_mono] {sequence}: already packed; skipping", flush=True)
        return True

    n_frames = _scan_frames(seq_mono_root, available_views, persons)
    if n_frames == 0:
        print(f"[pack_mono] {sequence}: no frames found", flush=True)
        return False

    n_views = len(views)  # keep ALL requested views (zeros for missing)
    view_to_idx = {str(v): i for i, v in enumerate(views)}
    print(
        f"[pack_mono] {sequence}: n_frames={n_frames} views={views} "
        f"available_views={available_views} persons={persons} workers={workers}",
        flush=True,
    )

    # Allocate output arrays per person.
    per_person = {}
    for person in persons:
        per_person[person] = {
            "k308": np.zeros((n_frames, n_views, 308, 3), dtype=np.float32),
            "k70": np.zeros((n_frames, n_views, 70, 3), dtype=np.float32),
            "shape": np.zeros((n_frames, n_views, 45), dtype=np.float32),
            "mp": np.zeros((n_frames, n_views, 204), dtype=np.float32),
        }

    # Enumerate all (path, frame, view_idx, person) tasks.
    tasks = []
    for v in available_views:
        vi = view_to_idx[str(v)]
        for person in persons:
            p_dir = os.path.join(seq_mono_root, str(v), person)
            if not os.path.isdir(p_dir):
                continue
            for fn in os.listdir(p_dir):
                if not fn.endswith(".npz") or fn.endswith(".tmp.npz"):
                    continue
                try:
                    fi = int(fn[:-4])
                except ValueError:
                    continue
                if fi >= n_frames:
                    continue
                tasks.append((os.path.join(p_dir, fn), fi, vi, person))

    print(f"[pack_mono] {sequence}: total tasks={len(tasks)}", flush=True)

    t0 = time.time()
    n_ok = 0
    n_fail = 0
    with mp.Pool(
        processes=max(1, workers),
        initializer=_init_worker,
        initargs=(sam3d_ckpt,),
    ) as pool:
        for result in pool.imap_unordered(_worker_call, tasks, chunksize=64):
            if result is None:
                n_fail += 1
                continue
            fi, vi, person, k308, k70, shape, mp_arr = result
            pp = per_person[person]
            pp["k308"][fi, vi] = k308
            pp["k70"][fi, vi] = k70
            pp["shape"][fi, vi] = shape
            pp["mp"][fi, vi] = mp_arr
            n_ok += 1

    dt = time.time() - t0
    print(
        f"[pack_mono] {sequence}: loaded {n_ok}/{len(tasks)} (fail={n_fail}) "
        f"in {dt:.1f}s",
        flush=True,
    )

    # Write atomically to tmp dir, then rename.
    tmp_dir = seq_out_dir + ".tmp"
    if os.path.exists(tmp_dir):
        import shutil

        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    views_arr = np.array([int(v) for v in views], dtype=np.int32)
    frame_indices = np.arange(n_frames, dtype=np.int32)
    for person in persons:
        p_out = os.path.join(tmp_dir, person)
        os.makedirs(p_out, exist_ok=True)
        pp = per_person[person]
        # Only persist k308 if any per-file mono actually contained 308-kp data.
        # Current HOI-M3 mono output stores 70-kp only -> k308 is all zeros; skip
        # writing to save ~76% of disk per sequence. Loader treats missing
        # k308.npy as zeros.
        if np.any(pp["k308"]):
            np.save(os.path.join(p_out, "keypoints2d_308.npy"), pp["k308"])
        np.save(os.path.join(p_out, "keypoints2d_70.npy"), pp["k70"])
        np.save(os.path.join(p_out, "shape_params.npy"), pp["shape"])
        np.save(os.path.join(p_out, "model_parameters.npy"), pp["mp"])
        np.savez(
            os.path.join(p_out, "meta.npz"),
            views=views_arr,
            frame_indices=frame_indices,
        )

    # Atomic swap (rename tmp -> seq_out_dir, replacing any existing).
    if os.path.exists(seq_out_dir):
        import shutil

        shutil.rmtree(seq_out_dir)
    os.replace(tmp_dir, seq_out_dir)

    # Sentinel.
    with open(os.path.join(seq_out_dir, ".pack_done"), "w") as f:
        f.write(f"n_frames={n_frames} persons={len(persons)} views={len(views)}\n")

    dt2 = time.time() - t0
    print(
        f"[pack_mono] {sequence}: WROTE packed cache in {dt2:.1f}s total",
        flush=True,
    )
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sequence", required=True, help="Sequence name")
    p.add_argument(
        "--mono_root",
        default="/scr/juze/datasets/HOI-M3/mhr_mono",
    )
    p.add_argument(
        "--output_root",
        default="/scr/juze/datasets/HOI-M3/mhr_mono_packed",
    )
    p.add_argument(
        "--views",
        nargs="+",
        default=["0", "2", "5", "6", "7", "8", "10", "11", "14", "15", "17", "19", "21", "22", "23", "24"],
    )
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--sam3d_ckpt", default=SAM3D_CKPT_DEFAULT)
    args = p.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    ok = pack_sequence(
        sequence=args.sequence,
        mono_root=args.mono_root,
        output_root=args.output_root,
        views=[str(v) for v in args.views],
        workers=args.workers,
        sam3d_ckpt=args.sam3d_ckpt,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
