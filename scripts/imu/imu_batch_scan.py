"""Batch IMU-vs-mocap_ground scan for all 52 livingroom seqs (steps 2+3 of the
correction pipeline; CPU-only). Per (seq, object with IMU json):
  - frame-free relative-angle comparison on a sliding grid (conjugation-invariant
    -> works with both world-frame and mount unknown)
  - segment classification: AGREE (<5 deg) / DIVERGE (>=5 deg)
  - RANSAC hand-eye A,B on agreeing pairs + inlier residual
Writes per-seq JSON reports to calibjoint/imu_scan/<seq>.json and a summary.
Objects needing pixel arbitration = those with any DIVERGE segment.
"""
import json, os, glob
import numpy as np

D = '/simurgh/group/juze/datasets/HOI-M3'
OUT = '/simurgh2/users/juze/calibjoint/imu_scan'
os.makedirs(OUT, exist_ok=True)
STEP = 150          # compare pairs (t, t+STEP)
AGREE_TH = 5.0      # deg


def ang(Ra, Rb):
    return np.degrees(np.arccos(np.clip((np.trace(Ra @ Rb.T) - 1) / 2, -1, 1)))


def proj_avg(Ms):
    U, _, Vt = np.linalg.svd(np.mean(Ms, 0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1; R = U @ Vt
    return R


def handeye_ransac(Rv, Ri, idx, iters=40, inlier_deg=5.0):
    """R_v[t] ~= A R_i[t] B. RANSAC over frame subsets, alternate A/B fits."""
    best = (None, None, 1e9, 0)
    rng = np.random.RandomState(0)
    for _ in range(iters):
        sub = rng.choice(idx, size=min(8, len(idx)), replace=False)
        A = np.eye(3); B = np.eye(3)
        for _ in range(15):
            A = proj_avg([Rv[t] @ (Ri[t] @ B).T for t in sub])
            B = proj_avg([(A @ Ri[t]).T @ Rv[t] for t in sub])
        res = np.array([ang(Rv[t], A @ Ri[t] @ B) for t in idx])
        inl = res < inlier_deg
        if inl.sum() >= max(6, 0.3 * len(idx)):
            A2 = proj_avg([Rv[t] @ (Ri[t] @ B).T for t in np.array(idx)[inl]])
            B2 = proj_avg([(A2 @ Ri[t]).T @ Rv[t] for t in np.array(idx)[inl]])
            res2 = np.array([ang(Rv[t], A2 @ Ri[t] @ B2) for t in np.array(idx)[inl]])
            if res2.mean() < best[2]:
                best = (A2, B2, float(res2.mean()), int(inl.sum()))
    return best


def scan_seq(seq):
    idir = f'{D}/imu_data/{seq}/imu_data'
    if not os.path.isdir(idir):
        idir = f'{D}/imu_data/{seq}'
    mgp = f'{D}/mocap_ground/{seq}_object.npz'
    if not os.path.isfile(mgp):
        return {'seq': seq, 'error': 'no mocap_ground'}
    mg = np.load(mgp, allow_pickle=True)['object_params']
    nmg = len(mg)
    report = {'seq': seq, 'objects': {}}
    for jf in sorted(glob.glob(f'{idir}/*.json')):
        name = os.path.basename(jf)[:-5]
        if '_fr' in name or any(c.isdigit() for c in name.split('chair')[-1] if 'chair' in name):
            continue
        if 'chair' in name:
            continue
        try:
            dj = json.load(open(jf))
            if not isinstance(dj, dict) or 'object_R' not in dj:
                continue
            Rj = np.array(dj['object_R']); s = int(dj['start_frame'])
        except Exception as e:
            report['objects'][name] = {'error': f'json:{e}'}; continue
        # mocap_ground entries: index i corresponds to video frame (mg_start + i);
        # assume same start offset family (mg starts at video_nf - nmg typically)
        e0 = mg[0]
        if name not in e0:
            report['objects'][name] = {'error': 'not in mocap_ground'}; continue
        mg_start = s  # empirically json start == mocap start (data03: both 560/21575+560)
        def Rmg(fr):
            i = fr - mg_start
            if i < 0 or i >= nmg: return None
            e = mg[i][name]
            e = e.item() if hasattr(e, 'item') else e
            return np.array(e['object_R'])
        n = len(Rj)
        pairs = []
        for a in range(0, n - STEP, STEP):
            fa, fb = s + a, s + a + STEP
            Ra_m, Rb_m = Rmg(fa), Rmg(fb)
            if Ra_m is None or Rb_m is None: continue
            ti = ang(Rj[a + STEP], Rj[a]); tm = ang(Rb_m, Ra_m)
            pairs.append((fa, ti, tm, abs(ti - tm)))
        if not pairs:
            report['objects'][name] = {'error': 'no overlapping pairs'}; continue
        diffs = np.array([p[3] for p in pairs])
        moved = np.array([max(p[1], p[2]) for p in pairs]) > 3.0  # pair where object moved
        div_pairs = [p for p in pairs if p[3] >= AGREE_TH]
        # divergence segments (merge consecutive)
        segs = []
        for fa, ti, tm, d in div_pairs:
            if segs and fa - segs[-1][1] <= STEP * 2:
                segs[-1][1] = fa + STEP
            else:
                segs.append([fa, fa + STEP])
        # hand-eye on agreeing MOVING pairs' endpoints (static frames are degenerate)
        agree_frames = [p[0] - s for p in pairs if p[3] < AGREE_TH]
        A = B = None; he_res = None; he_inl = 0
        if len(agree_frames) >= 8:
            Rv = {t: Rmg(s + t) for t in agree_frames}
            A, B, he_res, he_inl = handeye_ransac(Rv, Rj, agree_frames)
        report['objects'][name] = {
            'n_pairs': len(pairs), 'moved_pairs': int(moved.sum()),
            'diverging_pairs': len(div_pairs),
            'max_pair_diff_deg': float(diffs.max()),
            'divergence_segments': [[int(a), int(b)] for a, b in segs],
            'handeye_res_deg': he_res, 'handeye_inliers': he_inl,
            'A': A.tolist() if A is not None else None,
            'B': B.tolist() if B is not None else None,
        }
    return report


if __name__ == '__main__':
    seqs = sorted(os.listdir(f'{D}/imu_data'))
    summary = []
    for seq in seqs:
        try:
            r = scan_seq(seq)
        except Exception as e:
            r = {'seq': seq, 'error': f'{type(e).__name__}:{e}'}
        json.dump(r, open(f'{OUT}/{seq}.json', 'w'), indent=1)
        nobj = len(r.get('objects', {}))
        ndiv = sum(1 for o in r.get('objects', {}).values() if o.get('diverging_pairs', 0) > 0)
        print(f"{seq}: objects={nobj} with_divergence={ndiv} "
              f"{'ERR ' + r['error'] if 'error' in r else ''}", flush=True)
        summary.append((seq, nobj, ndiv))
    print('\n=== SUMMARY ===', flush=True)
    tot = sum(x[1] for x in summary); totd = sum(x[2] for x in summary)
    print(f'{len(summary)} seqs, {tot} imu-objects scanned, {totd} objects have divergence '
          f'(need pixel arbitration)', flush=True)
    print('BATCH_SCAN_DONE', flush=True)
