"""FAST swap-severity via HOIM3MaskSource (LZ4+threaded shard reads).
Reads mask_shards (what DenseFit reuse would actually consume) for a block of
consecutive frames; tracks each person's per-view centroid; counts ID-swaps
(centroid teleports >120px AND lands within 90px of another person's prior).
"""
import numpy as np, sys, os
sys.path.insert(0, '/simurgh/u/juze/code/HHOI-Toolkit/MvObjectFitting')
from utils.hoim3_loader import HOIM3MaskSource
VIEWS = [0, 7, 14, 21, 28]
NBLK = 200


def cen(m):
    ys, xs = np.where(m > 0)
    return None if len(xs) < 50 else np.array([xs.mean(), ys.mean()])


def scan(seq):
    ms = HOIM3MaskSource(seq)
    pers = ms.person_names
    if not pers:
        return None
    nv = ms.num_shard_views
    views = [v for v in VIEWS if v < nv]
    fids = ms.frame_ids
    start = len(fids) // 2
    blk = fids[start:start + min(NBLK, len(fids) - start)]
    prev = {}
    trans = swaps = bigjump = 0
    for fid in blk:
        d = ms.read_frames_parallel(pers, fid)  # {person: (nv,H,W)}
        cur = {}
        for v in views:
            cc = {p: cen(d[p][v]) for p in pers}
            for p in pers:
                cur[(v, p)] = cc[p]
                c, pc = cc[p], prev.get((v, p))
                if c is None or pc is None:
                    continue
                trans += 1
                if np.linalg.norm(c - pc) > 120:
                    bigjump += 1
                    for q in pers:
                        if q == p:
                            continue
                        oc = prev.get((v, q))
                        if oc is not None and np.linalg.norm(c - oc) < 90:
                            swaps += 1
                            break
        prev = cur
    ms.close()
    return dict(seq=seq, persons=len(pers), trans=trans, bigjump=bigjump, swaps=swaps,
                swap_rate=swaps / max(trans, 1))


if __name__ == '__main__':
    rows = []
    for s in sys.argv[1:]:
        try:
            r = scan(s)
            if r:
                rows.append(r)
                print(f"{r['seq']:22s} P{r['persons']} sw={r['swaps']:4d} "
                      f"jump={r['bigjump']:4d}/{r['trans']:5d} swap_rate={r['swap_rate']*100:5.2f}%",
                      flush=True)
        except Exception as e:
            print(f"{s:22s} ERR {type(e).__name__}: {str(e)[:50]}", flush=True)
    rows.sort(key=lambda r: -r['swap_rate'])
    print("\n=== RANKED worst-first ===", flush=True)
    for r in rows:
        tag = 'SEVERE' if r['swap_rate'] > 0.02 else 'moderate' if r['swap_rate'] > 0.008 else 'mild'
        print(f"  {r['seq']:22s} swap_rate={r['swap_rate']*100:5.2f}%  [{tag}]", flush=True)
