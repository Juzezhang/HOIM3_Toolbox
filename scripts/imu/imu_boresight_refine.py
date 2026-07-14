"""Visual boresight refinement: the hand-eye A,B was calibrated AGAINST
mocap_ground, whose rotations turn out to be IMU-derived for these objects —
so any constant orientation error is shared and invisible to that calibration.
Fix: refine small rotation deltas (dA on the world side, dB on the mount side)
directly against the IMAGES: maximize mean occlusion-aware IoU over K spread
frames, alternating with per-frame T refits.
  R_used(t) = (A·exp(dA)) · R_imu(t) · (exp(dB)·B)
Env: SEQ, OBJ. Prints before/after IoU and |dA|,|dB| in degrees.
"""
import os, json
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import numpy as np, cv2, trimesh, pyrender
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as SR

SEQ = os.environ.get('SEQ', 'livingroom_data04'); OBJ = os.environ.get('OBJ', 'bucket')
D = '/simurgh/group/juze/datasets/HOI-M3'
D2 = '/simurgh2/datasets/HOI-M3'
CT = f'{D2}/cutie_tracking/{SEQ}'
INFO = json.load(open('/simurgh/group/juze/datasets/HOI-M3/dataset_information.json'))
DATE = next(d for d, ss in INFO.items() if SEQ in ss)
scan = json.load(open(f'/simurgh2/users/juze/calibjoint/imu_scan/{SEQ}.json'))
so = scan['objects'][OBJ]
A0 = np.array(so['A']); B0 = np.array(so['B'])
dj = json.load(open(f'{D}/imu_data/{SEQ}/imu_data/{OBJ}.json'))
Rj = np.array(dj['object_R']); s = int(dj['start_frame'])
names = json.load(open(f'{D2}/cutie_refs/{SEQ}/masks/0_names.json'))['mask_names']
OBJ_IDX = names.index(OBJ) + 1
PERSON_IDX = tuple(i + 1 for i, n in enumerate(names) if n.startswith('person'))

cal = json.load(open(f'{D2}/calib_ground_refined/{DATE}/calibration.json'))
cams = {}
for v in [0, 2, 5, 7, 10, 14, 19, 21, 24, 29, 36]:
    c = cal.get(str(v))
    if not c or not c.get('K'): continue
    K = np.array(c['K'], float).reshape(3, 3) / 3.0; K[2, 2] = 1.0
    rt = np.array(c['RT'], float)
    RT = rt.reshape(4, 4)[:3] if rt.size == 16 else rt.reshape(3, 4)
    cams[v] = (K, RT[:, :3], RT[:, 3])
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

def masks_at(fr, cap=4):
    out = {}
    for v in cams:
        p = f'{CT}/{v}/{fr:06d}.npz'
        if not os.path.exists(p): continue
        m = np.load(p, allow_pickle=True)['mask']
        if m.shape != (720, 1280):
            m = cv2.resize(m, (1280, 720), interpolation=cv2.INTER_NEAREST)
        r = (m == OBJ_IDX)
        if r.sum() < 200: continue
        out[v] = (r, np.isin(m, PERSON_IDX))
    if len(out) > cap:
        keep = sorted(out, key=lambda v: -out[v][0].sum())[:cap]
        out = {v: out[v] for v in keep}
    return out

def tri_T(msk):
    rows = []
    for v, (r, _) in msk.items():
        ys, xs = np.where(r); u, w = xs.mean(), ys.mean()
        K, R, t = cams[v]; P = K @ np.hstack([R, t[:, None]])
        rows.append(u * P[2] - P[0]); rows.append(w * P[2] - P[1])
    M = np.array(rows); _, _, Vt = np.linalg.svd(M); X = Vt[-1]
    return X[:3] / X[3]

def occ_iou(R, T, msk):
    xs = []
    for v, (r, pe) in msk.items():
        S = sil(v, R, T); k = ~pe
        i = (S & r & k).sum(); u = ((S | r) & k).sum()
        if u > 100: xs.append(i / u)
    return float(np.mean(xs)) if xs else 0.0

def fit_T(R, msk, T0, fev=40):
    r = minimize(lambda T: -occ_iou(R, T, msk), T0, method='Nelder-Mead',
                 options={'maxfev': fev, 'xatol': 3e-3, 'fatol': 1e-3})
    return r.x, -r.fun

# ---- pick K spread frames with decent masks ----
K_FR = 14
n = len(Rj)
cand = list(range(s + 100, s + n - 100, max(1, n // 60)))
sel = []
for fr in cand:
    msk = masks_at(fr)
    if len(msk) >= 3:
        sel.append(fr)
    if len(sel) >= K_FR: break
print(f'{SEQ}/{OBJ}: refine frames {sel}', flush=True)

def Ruse(fr, dab):
    dA = SR.from_rotvec(dab[:3]).as_matrix(); dB = SR.from_rotvec(dab[3:]).as_matrix()
    return (A0 @ dA) @ Rj[fr - s] @ (dB @ B0)

state = {}  # fr -> (msk, T)
for fr in sel:
    msk = masks_at(fr)
    T0 = tri_T(msk)
    Tf, _ = fit_T(Ruse(fr, np.zeros(6)), msk, T0)
    state[fr] = (msk, Tf)

def score(dab):
    return -np.mean([occ_iou(Ruse(fr, dab), state[fr][1], state[fr][0]) for fr in sel])

base = -score(np.zeros(6))
print(f'baseline mean IoU (A,B as-is): {base:.4f}', flush=True)
dab = np.zeros(6)
for rnd in range(3):
    r = minimize(score, dab, method='Nelder-Mead',
                 options={'maxfev': 220, 'xatol': 1e-3, 'fatol': 5e-4})
    dab = r.x
    # re-fit T under new rotation
    for fr in sel:
        msk, T = state[fr]
        Tf, _ = fit_T(Ruse(fr, dab), msk, T, fev=30)
        state[fr] = (msk, Tf)
    cur = -score(dab)
    print(f'round {rnd+1}: mean IoU {cur:.4f}  |dA|={np.degrees(np.linalg.norm(dab[:3])):.2f}deg '
          f'|dB|={np.degrees(np.linalg.norm(dab[3:])):.2f}deg', flush=True)

print(f'\nRESULT {SEQ}/{OBJ}: IoU {base:.4f} -> {-score(dab):.4f}; '
      f'boresight correction |dA|={np.degrees(np.linalg.norm(dab[:3])):.2f}deg '
      f'|dB|={np.degrees(np.linalg.norm(dab[3:])):.2f}deg', flush=True)
out = dict(seq=SEQ, obj=OBJ, dA=dab[:3].tolist(), dB=dab[3:].tolist(),
           iou_before=float(base), iou_after=float(-score(dab)))
json.dump(out, open(f'/simurgh2/users/juze/calibjoint/boresight_{SEQ}_{OBJ}.json', 'w'))
print('BORESIGHT_DONE', flush=True)
