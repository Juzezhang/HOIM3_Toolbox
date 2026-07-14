"""Layer-3 PIXEL JUDGE: radio (livingroom_data03), IMU rotation vs mocap_ground
rotation on the divergence segment (fr>12000).

1. Solve constant A,B with R_mg(t) ~= A @ R_imu(t) @ B on the agreement segment
   (alternating chordal means; both trajectories agree to <3 deg there).
2. For each test frame and each R hypothesis {mocap_ground, IMU->ground}:
   fit T only (init = triangulated Cutie radio-mask centroids; Nelder-Mead on
   occlusion-aware IoU), render scanned radio mesh into views, report best IoU.
The hypothesis with higher IoU on divergence frames is the correct rotation.
Renders via pyrender EGL depth (silhouette = depth>0). calib_ground_refined
(pairs with mocap_ground), K/3 for 720p images/masks.
"""
import os, json
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import numpy as np, cv2, trimesh, pyrender
from scipy.optimize import minimize

D = '/simurgh/group/juze/datasets/HOI-M3'
D2 = '/simurgh2/datasets/HOI-M3'
CT = f'{D2}/cutie_tracking/livingroom_data03'
SEQ = 'livingroom_data03'; DATE = '20230912'
RADIO_IDX = 4; PERSON_IDX = (6, 7, 8)
VIEWS = [0, 2, 5, 7, 10, 14, 19, 24, 29, 36]
AGREE = list(range(600, 11000, 300))
TEST = [6000, 9000, 12500, 13500, 14500, 15500, 16500, 17500]

def ang(Ra, Rb):
    return np.degrees(np.arccos(np.clip((np.trace(Ra @ Rb.T) - 1) / 2, -1, 1)))

def proj_avg(Ms):
    U, _, Vt = np.linalg.svd(np.mean(Ms, 0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1; R = U @ Vt
    return R

# ---- load rotations ----
dj = json.load(open(f'{D}/imu_data/{SEQ}/imu_data/radio.json'))
Rj = np.array(dj['object_R']); Tj = np.array(dj['object_T']); s = dj['start_frame']
mg = np.load(f'{D}/mocap_ground/{SEQ}_object.npz', allow_pickle=True)['object_params']
def mg_entry(fr):
    e = mg[fr - 560]['radio']
    return e.item() if hasattr(e, 'item') else e
def Rmg(fr): return np.array(mg_entry(fr)['object_R'])
def Tmg(fr): return np.array(mg_entry(fr)['object_T']).ravel()

# ---- 1. hand-eye: R_mg ~= A R_imu B on agreement segment ----
A = np.eye(3); B = np.eye(3)
for it in range(30):
    A = proj_avg([Rmg(f) @ (Rj[f - s] @ B).T for f in AGREE])
    B = proj_avg([(A @ Rj[f - s]).T @ Rmg(f) for f in AGREE])
res = [ang(Rmg(f), A @ Rj[f - s] @ B) for f in AGREE]
print(f'hand-eye residual on agreement segment: mean={np.mean(res):.2f} deg  max={np.max(res):.2f} deg', flush=True)

# ---- calib + masks ----
cal = json.load(open(f'{D2}/calib_ground_refined/{DATE}/calibration.json'))
cams = {}
for v in VIEWS:
    c = cal[str(v)]
    K = np.array(c['K'], float).reshape(3, 3) / 3.0; K[2, 2] = 1.0
    rt = np.array(c['RT'], float)
    RT = rt.reshape(4, 4)[:3] if rt.size == 16 else rt.reshape(3, 4)
    cams[v] = (K, RT[:, :3], RT[:, 3])

def masks_at(fr):
    out = {}
    for v in VIEWS:
        p = f'{CT}/{v}/{fr:06d}.npz'
        if not os.path.exists(p): continue
        m = np.load(p, allow_pickle=True)['mask']
        if m.shape != (720, 1280):
            m = cv2.resize(m, (1280, 720), interpolation=cv2.INTER_NEAREST)
        radio = (m == RADIO_IDX)
        if radio.sum() < 200: continue
        pers = np.isin(m, PERSON_IDX)
        out[v] = (radio, pers)
    return out

def triangulate_T(msk):
    rows = []
    for v, (radio, _) in msk.items():
        ys, xs = np.where(radio)
        u, w = xs.mean(), ys.mean()
        K, R, t = cams[v]
        P = K @ np.hstack([R, t[:, None]])
        rows.append(u * P[2] - P[0]); rows.append(w * P[2] - P[1])
    Mx = np.array(rows)
    _, _, Vt = np.linalg.svd(Mx)
    X = Vt[-1]; return X[:3] / X[3]

# ---- mesh + renderer (METRIC variant: *_simplified_transformed.obj; the raw
# scan is non-metric — v3's auto-pick skips posed extents >15 m) ----
mesh_path = f'{D}/scanned_object/radio/radio_simplified_transformed.obj'
print('mesh:', mesh_path, flush=True)
tm = trimesh.load(mesh_path, force='mesh')
print('mesh extents(m):', np.round(tm.extents, 3), 'verts:', len(tm.vertices), flush=True)
ren = pyrender.OffscreenRenderer(1280, 720)
CV2GL = np.diag([1, -1, -1, 1.0])

def render_sil(v, R, T):
    K, Rc, tc = cams[v]
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0])
    pose = np.eye(4); pose[:3, :3] = R; pose[:3, 3] = T
    scene.add(pyrender.Mesh.from_trimesh(tm, smooth=False), pose=pose)
    cam = pyrender.IntrinsicsCamera(K[0, 0], K[1, 1], K[0, 2], K[1, 2], znear=0.05, zfar=20)
    E = np.eye(4); E[:3, :3] = Rc; E[:3, 3] = tc
    scene.add(cam, pose=np.linalg.inv(CV2GL @ E) @ np.eye(4))
    d = ren.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
    return d > 0

def occ_iou(fr, R, T, msk):
    ious = []
    for v, (radio, pers) in msk.items():
        sil = render_sil(v, R, T)
        keep = ~pers
        i = (sil & radio & keep).sum(); u = ((sil | radio) & keep).sum()
        if u > 100: ious.append(i / u)
    return float(np.mean(ious)) if ious else 0.0

def best_T_iou(fr, R, msk, T0):
    f = lambda T: -occ_iou(fr, R, T, msk)
    r = minimize(f, T0, method='Nelder-Mead',
                 options={'maxfev': 90, 'xatol': 3e-3, 'fatol': 1e-3})
    return -r.fun, r.x

print('\nframe   IoU_mocapR  IoU_imuR   winner   (Tinit from mask triangulation)', flush=True)
rows = []
for fr in TEST:
    msk = masks_at(fr)
    if len(msk) < 3:
        print(f'{fr}: radio visible in only {len(msk)} views, skip', flush=True); continue
    T0 = triangulate_T(msk)
    i_mg, _ = best_T_iou(fr, Rmg(fr), msk, T0)
    i_im, _ = best_T_iou(fr, A @ Rj[fr - s] @ B, msk, T0)
    win = 'mocap' if i_mg > i_im + 0.01 else ('IMU' if i_im > i_mg + 0.01 else 'tie')
    seg = 'AGREE' if fr < 12000 else 'DIVERGE'
    rows.append((fr, i_mg, i_im, win, seg))
    print(f'{fr:6d}  {i_mg:8.3f}  {i_im:8.3f}   {win:6s}  [{seg}] nviews={len(msk)}', flush=True)

print('\n=== VERDICT ===', flush=True)
div = [r for r in rows if r[4] == 'DIVERGE']
if div:
    mg_w = sum(1 for r in div if r[3] == 'mocap'); im_w = sum(1 for r in div if r[3] == 'IMU')
    print(f'divergence frames: mocap wins {mg_w}, IMU wins {im_w}, ties {len(div)-mg_w-im_w}', flush=True)
print('JUDGE_DONE', flush=True)
