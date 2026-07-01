"""Pack per-frame HOI-M3 mono MHR into a packed cache — TAR-AWARE variant.

Same output schema as pack_mono_cache.py, but for views whose mono is stored as
<seq>/<view>/data.tar (a tarball of ./personK/<frame>.npz) it reads members
straight from the tar instead of requiring extracted loose npz. Loose-npz views
are handled exactly as before. This avoids writing/reading tens of millions of
small npz files to NFS.

Per-view granularity: each (view) is one task. A loose view streams its dir; a
tarred view streams its tar. Workers return a list of per-frame records.
"""
import argparse
import io
import os
import sys
import tarfile
import time
import multiprocessing as mp

import numpy as np

sys.path.insert(0, "/simurgh/u/juze/code/mv-bodyfit")
from mvbodyfit.operations.mhr_sam3d import (
    build_mhr_model_parameters_from_sam3d_npz,
    load_sam3d_mhr_assets,
)

SAM3D_CKPT_DEFAULT = (
    "/simurgh/u/juze/code/fast-sam-3d-body/checkpoints/sam-3d-body-dinov3/model.ckpt"
)
_W_ASSETS = None


def _init_worker(sam3d_ckpt):
    global _W_ASSETS
    _W_ASSETS = None
    if sam3d_ckpt and os.path.exists(sam3d_ckpt):
        try:
            _W_ASSETS = load_sam3d_mhr_assets(sam3d_ckpt)
        except Exception as e:
            print(f"[pack_tar] worker asset load FAILED: {e}", flush=True)


def _parse_npz(data):
    """data: np.load result (NpzFile). Returns (k70, shape, mp)."""
    k308 = np.zeros((308, 3), dtype=np.float32)
    for src in ("pred_keypoints_2d_308", "keypoints2d_308"):
        if src in data.files:
            k = data[src]
            if k.shape == (308, 2):
                k308[:, :2] = k; k308[:, 2] = 1.0
            elif k.shape == (308, 3):
                k308 = k.astype(np.float32)
            break
    k70 = np.zeros((70, 3), dtype=np.float32)
    used = False
    for src in ("pred_keypoints_2d", "keypoints2d_70"):
        if src in data.files:
            k = data[src]
            if k.shape == (70, 2):
                k70[:, :2] = k; k70[:, 2] = 1.0
            elif k.shape == (70, 3):
                k70 = k.astype(np.float32)
            used = True
            break
    if not used:
        k70 = k308[:70].copy()
    shape = np.zeros(45, dtype=np.float32)
    if "shape_params" in data.files:
        s = data["shape_params"].flatten()[:45].astype(np.float32)
        shape[: len(s)] = s
    mp_arr = np.zeros(204, dtype=np.float32)
    if _W_ASSETS is not None and (
        "body_pose_params" in data.files
        or "hand_pose_params" in data.files
        or "scale_params" in data.files
    ):
        try:
            mp_arr = build_mhr_model_parameters_from_sam3d_npz(
                data, assets=_W_ASSETS, use_mono_global=False
            ).astype(np.float32)
        except Exception:
            mp_arr = np.zeros(204, dtype=np.float32)
    elif "model_parameters" in data.files:
        mpraw = np.asarray(data["model_parameters"], dtype=np.float32).reshape(-1)
        if mpraw.size >= 204:
            mp_arr = mpraw[:204]
        elif mpraw.size > 0:
            mp_arr[: mpraw.size] = mpraw
    return k70, shape, mp_arr


def _frame_from_name(name):
    base = os.path.basename(name)
    if not base.endswith(".npz") or base.endswith(".tmp.npz"):
        return None
    try:
        return int(base[:-4])
    except ValueError:
        return None


def _process_view(args):
    """One view. args=(view_idx, kind, path, n_frames).
    kind='dir' → path is person-parent dir; 'tar' → path is data.tar.
    Returns (view_idx, [(fi, person, k70, shape, mp), ...])."""
    view_idx, kind, path, n_frames = args
    out = []
    try:
        if kind == "tar":
            with tarfile.open(path, "r") as tf:
                for m in tf:
                    if not m.isfile():
                        continue
                    fi = _frame_from_name(m.name)
                    if fi is None or fi >= n_frames:
                        continue
                    # member path like ./person0/000999.npz
                    parts = m.name.strip("./").split("/")
                    person = next((p for p in parts if p.startswith("person")), None)
                    if person is None:
                        continue
                    f = tf.extractfile(m)
                    if f is None:
                        continue
                    try:
                        data = np.load(io.BytesIO(f.read()), allow_pickle=True)
                        k70, shape, mp_arr = _parse_npz(data)
                    except Exception:
                        continue  # skip a single corrupt npz, don't lose the view
                    out.append((fi, person, k70, shape, mp_arr))
        else:  # dir
            for person in os.listdir(path):
                p_dir = os.path.join(path, person)
                if not (person.startswith("person") and os.path.isdir(p_dir)):
                    continue
                for fn in os.listdir(p_dir):
                    fi = _frame_from_name(fn)
                    if fi is None or fi >= n_frames:
                        continue
                    data = np.load(os.path.join(p_dir, fn), allow_pickle=True)
                    k70, shape, mp_arr = _parse_npz(data)
                    out.append((fi, person, k70, shape, mp_arr))
    except Exception as e:
        print(f"[pack_tar] view {view_idx} ({path}) FAILED: {e}", flush=True)
    return view_idx, out


def _view_source(seq_mono_root, v):
    """Return ('dir', personparent) or ('tar', tarpath) or None."""
    v_dir = os.path.join(seq_mono_root, str(v))
    if not os.path.isdir(v_dir):
        return None
    # loose person dirs with npz?
    for entry in os.listdir(v_dir):
        if entry.startswith("person") and os.path.isdir(os.path.join(v_dir, entry)):
            # ensure it has at least one npz
            pd = os.path.join(v_dir, entry)
            for fn in os.listdir(pd):
                if fn.endswith(".npz"):
                    return ("dir", v_dir)
    tar = os.path.join(v_dir, "data.tar")
    if os.path.isfile(tar) and os.path.getsize(tar) > 0:
        return ("tar", tar)
    return None


def _scan_nframes(sources):
    """Find max frame index across all sources (cheap for dir; for tar, read names)."""
    max_fi = -1
    for kind, path in sources:
        if kind == "dir":
            for person in os.listdir(path):
                pd = os.path.join(path, person)
                if not os.path.isdir(pd):
                    continue
                for fn in os.listdir(pd):
                    fi = _frame_from_name(fn)
                    if fi is not None and fi > max_fi:
                        max_fi = fi
        else:
            with tarfile.open(path, "r") as tf:
                for m in tf:
                    fi = _frame_from_name(m.name)
                    if fi is not None and fi > max_fi:
                        max_fi = fi
    return max_fi + 1 if max_fi >= 0 else 0


def pack_sequence(sequence, mono_root, output_root, views, workers, sam3d_ckpt):
    seq_mono_root = os.path.join(mono_root, sequence)
    if not os.path.isdir(seq_mono_root):
        print(f"[pack_tar] {sequence}: mono dir missing", flush=True); return False
    view_to_idx = {str(v): i for i, v in enumerate(views)}
    # determine source per requested view
    view_src = {}
    for v in views:
        s = _view_source(seq_mono_root, v)
        if s is not None:
            view_src[str(v)] = s
    if not view_src:
        print(f"[pack_tar] {sequence}: no usable views", flush=True); return False
    # n_frames from a quick scan (prefer a dir source; else any)
    src_list = list(view_src.values())
    n_frames = _scan_nframes(src_list[:3] if len(src_list) >= 3 else src_list)
    if n_frames == 0:
        print(f"[pack_tar] {sequence}: no frames", flush=True); return False
    n_views = len(views)
    # discover persons from sources
    persons = set()
    for kind, path in src_list:
        if kind == "dir":
            for e in os.listdir(path):
                if e.startswith("person") and os.path.isdir(os.path.join(path, e)):
                    persons.add(e)
        else:
            with tarfile.open(path, "r") as tf:
                for m in tf:
                    parts = m.name.strip("./").split("/")
                    p = next((x for x in parts if x.startswith("person")), None)
                    if p:
                        persons.add(p)
                    if len(persons) >= 4:
                        break
    persons = sorted(persons, key=lambda x: int(x.replace("person", "")))
    print(f"[pack_tar] {sequence}: n_frames={n_frames} views={len(views)} "
          f"usable={len(view_src)} persons={persons} workers={workers} "
          f"(tar={sum(1 for k,_ in src_list if k=='tar')} dir={sum(1 for k,_ in src_list if k=='dir')})",
          flush=True)
    per_person = {p: {
        "k70": np.zeros((n_frames, n_views, 70, 3), dtype=np.float32),
        "shape": np.zeros((n_frames, n_views, 45), dtype=np.float32),
        "mp": np.zeros((n_frames, n_views, 204), dtype=np.float32),
    } for p in persons}
    tasks = [(view_to_idx[v], kind, path, n_frames) for v, (kind, path) in view_src.items()]
    t0 = time.time()
    n_rec = 0
    with mp.Pool(max(1, workers), initializer=_init_worker, initargs=(sam3d_ckpt,)) as pool:
        for view_idx, recs in pool.imap_unordered(_process_view, tasks):
            for fi, person, k70, shape, mp_arr in recs:
                if person not in per_person:
                    continue
                pp = per_person[person]
                pp["k70"][fi, view_idx] = k70
                pp["shape"][fi, view_idx] = shape
                pp["mp"][fi, view_idx] = mp_arr
                n_rec += 1
    print(f"[pack_tar] {sequence}: loaded {n_rec} records in {time.time()-t0:.1f}s", flush=True)
    tmp_dir = os.path.join(output_root, sequence) + ".tmp"
    if os.path.exists(tmp_dir):
        import shutil; shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)
    views_arr = np.array([int(v) for v in views], dtype=np.int32)
    frame_indices = np.arange(n_frames, dtype=np.int32)
    for person in persons:
        p_out = os.path.join(tmp_dir, person); os.makedirs(p_out, exist_ok=True)
        pp = per_person[person]
        np.save(os.path.join(p_out, "keypoints2d_70.npy"), pp["k70"])
        np.save(os.path.join(p_out, "shape_params.npy"), pp["shape"])
        np.save(os.path.join(p_out, "model_parameters.npy"), pp["mp"])
        np.savez(os.path.join(p_out, "meta.npz"), views=views_arr, frame_indices=frame_indices)
    seq_out_dir = os.path.join(output_root, sequence)
    if os.path.exists(seq_out_dir):
        import shutil; shutil.rmtree(seq_out_dir)
    os.replace(tmp_dir, seq_out_dir)
    with open(os.path.join(seq_out_dir, ".pack_done"), "w") as f:
        f.write(f"n_frames={n_frames} persons={len(persons)} views={len(views)}\n")
    print(f"[pack_tar] {sequence}: WROTE in {time.time()-t0:.1f}s total", flush=True)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sequence", required=True)
    p.add_argument("--mono_root", default="/simurgh2/datasets/HOI-M3/mhr_mono")
    p.add_argument("--output_root", default="/simurgh2/datasets/HOI-M3/mhr_mono_packed")
    p.add_argument("--views", nargs="+", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--sam3d_ckpt", default=SAM3D_CKPT_DEFAULT)
    a = p.parse_args()
    os.makedirs(a.output_root, exist_ok=True)
    ok = pack_sequence(a.sequence, a.mono_root, a.output_root,
                       [str(v) for v in a.views], a.workers, a.sam3d_ckpt)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
