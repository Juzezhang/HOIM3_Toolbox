"""Merge an existing (non-37) packed cache with the MISSING canonical views to produce
a 37-view pack — WITHOUT re-reading the views the old pack already has.

Old packs are mostly 16-view (loose-npz views). The 21 missing canonical views are
mostly data.tar (fast sequential) + a few loose. Reusing the old pack's arrays avoids
re-reading ~28 loose views (600k small NFS files) — the I/O bottleneck.

For each person: load old k70/shape/mp arrays, pack only the missing views (tar-aware),
slot everything into 37-view arrays by canonical view index + frame index, write.
"""
import argparse, os, sys, time
import numpy as np
sys.path.insert(0, "/simurgh/u/juze/code/HOIM3_Toolbox/scripts/regen")
# reuse the tar-aware per-view loader + asset init
import pack_mono_cache_tar as pt

CANON = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,34,35,36,37,38,39,41]


def merge_one(seq, mono_root, pack_root, sam3d_ckpt, workers):
    seqdir = os.path.join(pack_root, seq)
    persons = sorted([p for p in os.listdir(seqdir) if p.startswith('person') and
                      os.path.isdir(os.path.join(seqdir, p))]) if os.path.isdir(seqdir) else []
    if not persons:
        print(f"[merge] {seq}: no old pack persons", flush=True); return False
    # validate the old pack is complete: every person needs all 4 canonical files and
    # no interrupted-GCS-download .gstmp temp files. If not, bail so a full repack handles it
    # (the 6 partial-download seqs bedroom_data12/16/18/21/22/25 hit this).
    REQ = ('keypoints2d_70.npy', 'shape_params.npy', 'model_parameters.npy', 'meta.npz')
    for p in persons:
        names = set(os.listdir(os.path.join(seqdir, p)))
        if any(n.endswith('.gstmp') for n in names):
            print(f"[merge] {seq}: {p} has .gstmp (partial download) — SKIP merge, needs full pack", flush=True); return False
        miss = [r for r in REQ if r not in names]
        if miss:
            print(f"[merge] {seq}: {p} missing {miss} — SKIP merge, needs full pack", flush=True); return False
    # old views (from person0 meta)
    meta0 = np.load(os.path.join(seqdir, persons[0], 'meta.npz'))
    old_views = [int(v) for v in meta0['views']]
    if old_views == CANON:
        print(f"[merge] {seq}: already 37-view", flush=True); return True
    n_frames = int(meta0['frame_indices'].shape[0])
    missing = [v for v in CANON if v not in old_views]
    print(f"[merge] {seq}: old={len(old_views)}v n_frames={n_frames} missing={len(missing)}v {missing}", flush=True)

    # source for each missing view (dir/tar) from NFS
    seq_mono = os.path.join(mono_root, seq)
    miss_src = {}
    for v in missing:
        s = pt._view_source(seq_mono, v)
        if s is not None:
            miss_src[v] = s
    print(f"[merge] {seq}: missing views with data: {sorted(miss_src.keys())} "
          f"(tar={sum(1 for k,_ in miss_src.values() if k=='tar')} dir={sum(1 for k,_ in miss_src.values() if k=='dir')})", flush=True)

    # pack the missing views (tar-aware), per person
    cidx = {v: i for i, v in enumerate(CANON)}
    t0 = time.time()
    # load missing-view records via pool
    import multiprocessing as mp
    tasks = [(v, kind, path, n_frames) for v, (kind, path) in miss_src.items()]
    # per-person new arrays
    new = {p: {'k70': np.zeros((n_frames, len(missing), 70, 3), np.float32),
               'shape': np.zeros((n_frames, len(missing), 45), np.float32),
               'mp': np.zeros((n_frames, len(missing), 204), np.float32)} for p in persons}
    midx = {v: i for i, v in enumerate(missing)}
    with mp.Pool(max(1, workers), initializer=pt._init_worker, initargs=(sam3d_ckpt,)) as pool:
        # _process_view uses view_idx as first elem; pass the missing-array index
        remap = [(midx[v], kind, path, n_frames) for (v, kind, path, n_frames) in tasks]
        for vi, recs in pool.imap_unordered(pt._process_view, remap):
            for fi, person, k70, shape, mp_arr in recs:
                if person not in new: continue
                pp = new[person]; pp['k70'][fi, vi] = k70; pp['shape'][fi, vi] = shape; pp['mp'][fi, vi] = mp_arr
    print(f"[merge] {seq}: packed missing views in {time.time()-t0:.1f}s", flush=True)

    # assemble 37-view per person + write
    tmp = seqdir + '.mtmp'
    if os.path.exists(tmp):
        import shutil; shutil.rmtree(tmp)
    os.makedirs(tmp, exist_ok=True)
    for p in persons:
        old_k70 = np.load(os.path.join(seqdir, p, 'keypoints2d_70.npy'), mmap_mode='r')
        old_sh  = np.load(os.path.join(seqdir, p, 'shape_params.npy'), mmap_mode='r')
        old_mp  = np.load(os.path.join(seqdir, p, 'model_parameters.npy'), mmap_mode='r')
        of = old_k70.shape[0]
        F = min(of, n_frames)
        out_k70 = np.zeros((n_frames, 37, 70, 3), np.float32)
        out_sh  = np.zeros((n_frames, 37, 45), np.float32)
        out_mp  = np.zeros((n_frames, 37, 204), np.float32)
        oidx = {v: i for i, v in enumerate(old_views)}
        for v in CANON:
            ci = cidx[v]
            if v in oidx:
                oi = oidx[v]
                out_k70[:F, ci] = old_k70[:F, oi]; out_sh[:F, ci] = old_sh[:F, oi]; out_mp[:F, ci] = old_mp[:F, oi]
            elif v in midx:
                mi = midx[v]
                out_k70[:, ci] = new[p]['k70'][:, mi]; out_sh[:, ci] = new[p]['shape'][:, mi]; out_mp[:, ci] = new[p]['mp'][:, mi]
        po = os.path.join(tmp, p); os.makedirs(po, exist_ok=True)
        np.save(os.path.join(po, 'keypoints2d_70.npy'), out_k70)
        np.save(os.path.join(po, 'shape_params.npy'), out_sh)
        np.save(os.path.join(po, 'model_parameters.npy'), out_mp)
        np.savez(os.path.join(po, 'meta.npz'), views=np.array(CANON, np.int32),
                 frame_indices=np.arange(n_frames, np.int32) if False else np.arange(n_frames, dtype=np.int32))
    import shutil, gc
    # release the mmap'd old-pack arrays BEFORE touching seqdir — on NFS, unlinking a
    # file that's still mmap'd triggers a .nfsXXXX silly-rename, leaving the dir
    # "not empty" and rmdir/rmtree failing (OSError 39). del + gc closes the maps.
    try:
        del old_k70, old_sh, old_mp
    except NameError:
        pass
    gc.collect()
    # NFS-safe swap: rename old pack aside (no in-place unlink), put new in place,
    # then best-effort delete the aside copy (.nfs* residue clears on process exit).
    old_aside = seqdir + '.old'
    if os.path.exists(old_aside):
        shutil.rmtree(old_aside, ignore_errors=True)
    os.rename(seqdir, old_aside)
    os.replace(tmp, seqdir)
    shutil.rmtree(old_aside, ignore_errors=True)
    with open(os.path.join(seqdir, '.pack_done'), 'w') as f:
        f.write(f"merged37 n_frames={n_frames} persons={len(persons)}\n")
    print(f"[merge] {seq}: WROTE 37-view merged pack in {time.time()-t0:.1f}s total", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sequence', required=True)
    ap.add_argument('--mono_root', default='/simurgh2/datasets/HOI-M3/mhr_mono')
    ap.add_argument('--pack_root', default='/simurgh2/datasets/HOI-M3/mhr_mono_packed')
    ap.add_argument('--sam3d_ckpt', default=pt.SAM3D_CKPT_DEFAULT)
    ap.add_argument('--workers', type=int, default=12)
    a = ap.parse_args()
    ok = merge_one(a.sequence, a.mono_root, a.pack_root, a.sam3d_ckpt, a.workers)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
