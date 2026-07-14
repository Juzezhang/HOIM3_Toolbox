"""PRODUCTION object-pose fusion (recipe v2, QC-gated). Env: SEQ, OBJ.
1. QC pass (stride 30): occlusion-aware IoU of the mocap_ground pose.
2. BAD segments = QC-IoU < 0.35 OR inside scan divergence segments (dilated/merged).
3. Inside BAD segments: R = A*R_imu*B (hand-eye), T refit visually (stride 5,
   Nelder-Mead 70 evals, warm-start chain), savgol(9) smooth, linear interp between.
   Frames without >=2 mask views: T interpolated (source flag 2).
4. Outside: keep mocap pose (source 0).
Output: /simurgh2/datasets/HOI-M3/object_imu_fused/<seq>/<obj>.npz
  object_R (N,3,3), object_T (N,3), source (N,) uint8 {0 mocap,1 imu+fitT,2 imu+interpT},
  start_frame (video frame of index 0 == imu json start_frame)
Never touches object/ or mocap_ground/.
"""
import os, json, sys
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import numpy as np, cv2, trimesh, pyrender
from scipy.optimize import minimize
from scipy.signal import savgol_filter

SEQ = os.environ['SEQ']; OBJ = os.environ['OBJ']
D = '/simurgh/group/juze/datasets/HOI-M3'
D2 = '/simurgh2/datasets/HOI-M3'
OUTD = f'{D2}/object_imu_fused/{SEQ}'
os.makedirs(OUTD, exist_ok=True)
INFO = json.load(open(f'{D}/dataset_information.json'))
DATE = next(d for d, ss in INFO.items() if SEQ in ss)
scan = json.load(open(f'/simurgh2/users/juze/calibjoint/imu_scan/{SEQ}.json'))
so = scan['objects'][OBJ]
A = np.array(so['A']); B = np.array(so['B']); DIV = so.get('divergence_segments', [])
dj = json.load(open(f'{D}/imu_data/{SEQ}/imu_data/{OBJ}.json'))
Rj = np.array(dj['object_R']); s = int(dj['start_frame'])
mg = np.load(f'{D}/mocap_ground/{SEQ}_object.npz', allow_pickle=True)['object_params']
N = min(len(Rj), len(mg))

def mgp(i):
    e = mg[i]
    if not isinstance(e, dict) or OBJ not in e: return None, None
    e = e[OBJ]; e = e.item() if hasattr(e, 'item') else e
    return np.array(e['object_R']), np.array(e['object_T']).ravel()

cal = json.load(open(f'{D2}/calib_ground_refined/{DATE}/calibration.json'))
cams = {}
for v in [0, 2, 5, 7, 10, 14, 19, 21, 24, 29, 36]:
    c = cal.get(str(v))
    if not c or not c.get('K'): continue
    K = np.array(c['K'], float).reshape(3, 3) / 3.0; K[2, 2] = 1.0
    rt = np.array(c['RT'], float)
    RT = rt.reshape(4, 4)[:3] if rt.size == 16 else rt.reshape(3, 4)
    cams[v] = (K, RT[:, :3], RT[:, 3])

# ---- mask source: cutie_tracking npz if available, else LZ4 shards ----
CT = f'{D2}/cutie_tracking/{SEQ}'
USE_CUTIE = os.path.isfile(f'{CT}/0/.tracked_done')
if USE_CUTIE:
    names = json.load(open(f'{D2}/cutie_refs/{SEQ}/masks/0_names.json'))['mask_names']
    OBJ_IDX = names.index(OBJ) + 1
    PERSON_IDX = tuple(i + 1 for i, n in enumerate(names) if n.startswith('person'))
    def raw_masks(fr):
        out = {}
        for v in cams:
            p = f'{CT}/{v}/{fr:06d}.npz'
            if not os.path.exists(p): continue
            m = np.load(p, allow_pickle=True)['mask']
            out[v] = ((m == OBJ_IDX), np.isin(m, PERSON_IDX))
        return out
else:
    sys.path.insert(0, '/simurgh/u/juze/code/HHOI-Toolkit/MvObjectFitting')
    from utils.hoim3_loader import HOIM3MaskSource
    ms = HOIM3MaskSource(SEQ)
    assert OBJ in ms.all_names, f'{OBJ} not in shards'
    PNAMES = ms.person_names
    def raw_masks(fr):
        d = ms.read_frames_parallel([OBJ] + PNAMES, fr)
        out = {}
        for v in cams:
            if v >= ms.num_shard_views: continue
            om = d[OBJ][v] > 0
            pm = np.zeros_like(om)
            for pn in PNAMES: pm |= (d[pn][v] > 0)
            out[v] = (om, pm)
        return out

def masks_at(fr, cap=4):
    out = {}
    for v, (om, pm) in raw_masks(fr).items():
        if om.shape != (720, 1280):
            om = cv2.resize(om.astype(np.uint8), (1280, 720), interpolation=cv2.INTER_NEAREST) > 0
            pm = cv2.resize(pm.astype(np.uint8), (1280, 720), interpolation=cv2.INTER_NEAREST) > 0
        if om.sum() < 150: continue
        out[v] = (om, pm)
    if len(out) > cap:
        keep = sorted(out, key=lambda v: -out[v][0].sum())[:cap]
        out = {v: out[v] for v in keep}
    return out

tm = trimesh.load(f'{D}/scanned_object/{OBJ}/{OBJ}_simplified_transformed.obj', force='mesh')
ren = pyrender.OffscreenRenderer(1280, 720)
CV2GL = np.diag([1, -1, -1, 1.0])
mesh = pyrender.Mesh.from_trimesh(tm, smooth=False)

def sil(v, R, T):
    K, Rc, tc = cams[v]
    sc = pyrender.Scene(bg_color=[0, 0, 0, 0])
    p = np.eye(4); p[:3, :3] = R; p[:3, 3] = T; sc.add(mesh, pose=p)
    cam = pyrender.IntrinsicsCamera(K[0, 0], K[1, 1], K[0, 2], K[1, 2], znear=0.05, zfar=20)
    E = np.eye(4); E[:3, :3] = Rc; E[:3, 3] = tc
    sc.add(cam, pose=np.linalg.inv(CV2GL @ E))
    return ren.render(sc, flags=pyrender.RenderFlags.DEPTH_ONLY) > 0

def occ_iou(R, T, msk):
    xs = []
    for v, (r, pe) in msk.items():
        S = sil(v, R, T); k = ~pe
        i = (S & r & k).sum(); u = ((S | r) & k).sum()
        if u > 100: xs.append(i / u)
    return float(np.mean(xs)) if xs else None

def tri_T(msk):
    rows = []
    for v, (r, _) in msk.items():
        ys, xs = np.where(r); u, w = xs.mean(), ys.mean()
        K, R, t = cams[v]; P = K @ np.hstack([R, t[:, None]])
        rows.append(u * P[2] - P[0]); rows.append(w * P[2] - P[1])
    M = np.array(rows); _, _, Vt = np.linalg.svd(M); X = Vt[-1]
    return X[:3] / X[3]

# ---- 1. QC pass ----
QC_STRIDE = 30; QC_TH = 0.35
qc = {}
for i in range(0, N, QC_STRIDE):
    Rm, Tm = mgp(i)
    if Rm is None: qc[i] = -1; continue
    msk = masks_at(s + i)
    if len(msk) < 2: qc[i] = -2; continue   # object not visible: leave mocap
    v = occ_iou(Rm, Tm, msk)
    qc[i] = v if v is not None else -2
bad_idx = set()
for i, v in qc.items():
    if v == -1 or (0 <= v < QC_TH):
        bad_idx.add(i)
for a, b in DIV:
    for i in range(max(0, a - s), min(N, b - s), QC_STRIDE):
        bad_idx.add(i - i % QC_STRIDE)
# dilate +-45 and merge into segments
marks = np.zeros(N, bool)
for i in bad_idx:
    marks[max(0, i - 45):min(N, i + QC_STRIDE + 45)] = True
segs = []
i = 0
while i < N:
    if marks[i]:
        j = i
        while j < N and (marks[j] or (j - i < 90)): j += 1
        segs.append([i, j]); i = j
    else: i += 1
frac = sum(b - a for a, b in segs) / max(N, 1)
print(f'{SEQ}/{OBJ}: N={N} bad segments={len(segs)} covering {frac*100:.0f}%  '
      f'(QC med={np.median([v for v in qc.values() if v>=0]) if any(v>=0 for v in qc.values()) else -1:.3f})', flush=True)

# ---- 2. build fused track ----
R_out = np.zeros((N, 3, 3), np.float32); T_out = np.zeros((N, 3), np.float32)
src = np.zeros(N, np.uint8)
for i in range(N):
    Rm, Tm = mgp(i)
    if Rm is None:
        R_out[i] = A @ Rj[i] @ B; T_out[i] = T_out[i - 1] if i else 0; src[i] = 2
    else:
        R_out[i] = Rm; T_out[i] = Tm
FIT_STRIDE = 5 if frac < 0.6 else 8
for a, b in segs:
    fit_i, fit_T = [], []
    prev = None
    for i in range(a, b, FIT_STRIDE):
        msk = masks_at(s + i)
        Rimu = A @ Rj[i] @ B
        if len(msk) < 2:
            continue
        T0 = tri_T(msk)
        if prev is not None and (occ_iou(Rimu, prev, msk) or 0) > (occ_iou(Rimu, T0, msk) or 0):
            T0 = prev
        r = minimize(lambda T: -(occ_iou(Rimu, T, msk) or 0), T0, method='Nelder-Mead',
                     options={'maxfev': 70, 'xatol': 3e-3, 'fatol': 1e-3})
        prev = r.x
        fit_i.append(i); fit_T.append(r.x)
    if len(fit_T) >= 3:
        Tarr = np.array(fit_T)
        if len(Tarr) >= 9:
            Tarr = savgol_filter(Tarr, 9, 2, axis=0)
        # fill segment: R=IMU everywhere; T interp from fitted knots
        for i in range(a, b):
            R_out[i] = A @ Rj[i] @ B
            T_out[i] = np.array([np.interp(i, fit_i, Tarr[:, k]) for k in range(3)])
            src[i] = 1 if any(abs(i - fi) < FIT_STRIDE for fi in fit_i) else 2
    else:
        # object not visible enough here: keep IMU R, carry/mocap T, flag 2
        for i in range(a, b):
            R_out[i] = A @ Rj[i] @ B; src[i] = 2
print(f'  fitted segments done; src counts: mocap={int((src==0).sum())} fitT={int((src==1).sum())} interp={int((src==2).sum())}', flush=True)

# ---- 3. QC after (on bad-segment QC samples) ----
before, after = [], []
for i, v in qc.items():
    if not (0 <= v < QC_TH): continue
    msk = masks_at(s + i)
    if len(msk) < 2: continue
    va = occ_iou(R_out[i], T_out[i], msk)
    if va is not None: before.append(v); after.append(va)
if before:
    print(f'  QC on bad samples: mocap IoU {np.mean(before):.3f} -> fused {np.mean(after):.3f} (n={len(before)})', flush=True)

np.savez(f'{OUTD}/{OBJ}.npz', object_R=R_out, object_T=T_out, source=src,
         start_frame=np.int32(s))
json.dump(dict(seq=SEQ, obj=OBJ, start_frame=s, n=N, A=A.tolist(), B=B.tolist(),
               bad_segments=[[int(a), int(b)] for a, b in segs], bad_frac=float(frac),
               qc_before=float(np.mean(before)) if before else None,
               qc_after=float(np.mean(after)) if before else None,
               mask_source='cutie' if USE_CUTIE else 'shards'),
          open(f'{OUTD}/{OBJ}_meta.json', 'w'))
print(f'FUSE_DONE {SEQ}/{OBJ}', flush=True)
