"""Measure the small rotation offset the eye sees but silhouette IoU cannot:
photometric (textured-render NCC) 6-DoF refinement from the IMU pose, on static
plateaus (distinct orientations). Per-frame delta-rotation dR is reported; a
consistent mean dR across plateaus = true boresight correction (silhouette-blind).
Env: SEQ, OBJ.
"""
import os, json
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import numpy as np, cv2, trimesh, pyrender
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as SR

SEQ = os.environ.get('SEQ', 'livingroom_data03'); OBJ = os.environ.get('OBJ', 'radio')
D = '/simurgh/group/juze/datasets/HOI-M3'
D2 = '/simurgh2/datasets/HOI-M3'
CT = f'{D2}/cutie_tracking/{SEQ}'
INFO = json.load(open(f'{D}/dataset_information.json'))
DATE = next(d for d, ss in INFO.items() if SEQ in ss)
scan = json.load(open(f'/simurgh2/users/juze/calibjoint/imu_scan/{SEQ}.json'))
so = scan['objects'][OBJ]
A = np.array(so['A']); B = np.array(so['B'])
dj = json.load(open(f'{D}/imu_data/{SEQ}/imu_data/{OBJ}.json'))
Rj = np.array(dj['object_R']); s = int(dj['start_frame'])
names = json.load(open(f'{D2}/cutie_refs/{SEQ}/masks/0_names.json'))['mask_names']
OBJ_IDX = names.index(OBJ) + 1
PERSON_IDX = tuple(i + 1 for i, n in enumerate(names) if n.startswith('person'))

def ang(Ra, Rb):
    return np.degrees(np.arccos(np.clip((np.trace(Ra @ Rb.T) - 1) / 2, -1, 1)))

cal = json.load(open(f'{D2}/calib_ground_refined/{DATE}/calibration.json'))
cams = {}
for v in [0, 2, 5, 7, 10, 14, 19, 21, 24, 29, 36]:
    c = cal.get(str(v))
    if not c or not c.get('K'): continue
    K = np.array(c['K'], float).reshape(3, 3) / 3.0; K[2, 2] = 1.0
    rt = np.array(c['RT'], float)
    RT = rt.reshape(4, 4)[:3] if rt.size == 16 else rt.reshape(3, 4)
    cams[v] = (K, RT[:, :3], RT[:, 3])

# textured mesh (metric variant); pyrender keeps the OBJ texture material
mesh_path = f'{D}/scanned_object/{OBJ}/{OBJ}_simplified_transformed.obj'
tm = trimesh.load(mesh_path, force='mesh')
mesh = pyrender.Mesh.from_trimesh(tm, smooth=False)
ren = pyrender.OffscreenRenderer(1280, 720)
CV2GL = np.diag([1, -1, -1, 1.0])

def render_rgb_depth(v, R, T):
    K, Rc, tc = cams[v]
    sc = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[1.0, 1.0, 1.0])
    p = np.eye(4); p[:3, :3] = R; p[:3, 3] = T
    sc.add(mesh, pose=p)
    cam = pyrender.IntrinsicsCamera(K[0, 0], K[1, 1], K[0, 2], K[1, 2], znear=0.05, zfar=20)
    E = np.eye(4); E[:3, :3] = Rc; E[:3, 3] = tc
    sc.add(cam, pose=np.linalg.inv(CV2GL @ E))
    col, dep = ren.render(sc, flags=pyrender.RenderFlags.FLAT)
    return col, dep > 0

def masks_at(fr, cap=3):
    out = {}
    for v in cams:
        p = f'{CT}/{v}/{fr:06d}.npz'
        if not os.path.exists(p): continue
        m = np.load(p, allow_pickle=True)['mask']
        if m.shape != (720, 1280):
            m = cv2.resize(m, (1280, 720), interpolation=cv2.INTER_NEAREST)
        r = (m == OBJ_IDX)
        if r.sum() < 400: continue
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

def photo_score(R, T, msk, imgs):
    """masked grayscale NCC between FLAT textured render and the real image."""
    scores = []
    for v, (r, pe) in msk.items():
        col, sil = render_rgb_depth(v, R, T)
        m = sil & (~pe)
        if m.sum() < 300: continue
        a = cv2.cvtColor(col, cv2.COLOR_RGB2GRAY)[m].astype(np.float32)
        b = cv2.cvtColor(imgs[v], cv2.COLOR_BGR2GRAY)[m].astype(np.float32)
        a -= a.mean(); b -= b.mean()
        d = np.linalg.norm(a) * np.linalg.norm(b)
        if d < 1e-6: continue
        scores.append(float(a @ b / d))
    return float(np.mean(scores)) if scores else -1.0

def iou_score(R, T, msk):
    xs = []
    for v, (r, pe) in msk.items():
        _, S = render_rgb_depth(v, R, T)
        k = ~pe
        i = (S & r & k).sum(); u = ((S | r) & k).sum()
        if u > 100: xs.append(i / u)
    return float(np.mean(xs)) if xs else 0.0

# ---- static plateaus: |rel rot over 40fr| < 0.5deg, spaced apart ----
plateaus = []
step = 40
for i in range(200, len(Rj) - step - 200, step):
    if ang(Rj[i + step], Rj[i]) < 0.5:
        fr = s + i
        if not plateaus or fr - plateaus[-1] > 1500:
            plateaus.append(fr)
    if len(plateaus) >= 8: break
print(f'{SEQ}/{OBJ}: plateau frames {plateaus}', flush=True)

results = []
for fr in plateaus:
    msk = masks_at(fr)
    imgs = {v: cv2.imread(f'{D2}/images/{SEQ}/{v}/{fr:06d}.jpg') for v in msk}
    if len(msk) < 2 or any(im is None for im in imgs.values()):
        continue
    R0 = A @ Rj[fr - s] @ B
    # T init: IoU fit (silhouette centers it)
    r0 = minimize(lambda T: -iou_score(R0, T, msk), tri_T(msk), method='Nelder-Mead',
                  options={'maxfev': 60, 'xatol': 3e-3})
    T0 = r0.x
    ncc0 = photo_score(R0, T0, msk, imgs)
    # photometric 6-dof refine
    def obj(x):
        dR = SR.from_rotvec(x[:3]).as_matrix()
        return -photo_score(R0 @ dR, T0 + x[3:], msk, imgs)
    rr = minimize(obj, np.zeros(6), method='Nelder-Mead',
                  options={'maxfev': 260, 'xatol': 5e-4, 'fatol': 1e-4})
    drot = np.degrees(np.linalg.norm(rr.x[:3]))
    axis = rr.x[:3] / (np.linalg.norm(rr.x[:3]) + 1e-12)
    ncc1 = -rr.fun
    results.append((fr, drot, axis, ncc0, ncc1, rr.x[:3]))
    print(f'  fr{fr}: dR={drot:5.2f}deg axis=[{axis[0]:+.2f},{axis[1]:+.2f},{axis[2]:+.2f}] '
          f'NCC {ncc0:.3f}->{ncc1:.3f}', flush=True)

if results:
    vecs = np.array([r[5] for r in results])
    mean_vec = vecs.mean(0)
    mean_mag = np.degrees(np.linalg.norm(mean_vec))
    per = np.degrees(np.linalg.norm(vecs - mean_vec, axis=1))
    print(f'\nSUMMARY {SEQ}/{OBJ}: per-plateau dR mean magnitude '
          f'{np.mean([r[1] for r in results]):.2f}deg; CONSISTENT component '
          f'{mean_mag:.2f}deg (residual scatter {per.mean():.2f}deg)', flush=True)
    print('=> consistent component = true boresight to add to B; scatter = per-pose/other error', flush=True)
    json.dump(dict(seq=SEQ, obj=OBJ, mean_rotvec=mean_vec.tolist(),
                   mean_mag_deg=float(mean_mag), scatter_deg=float(per.mean()),
                   per_frame=[dict(fr=int(r[0]), mag=float(r[1]), ncc0=float(r[3]), ncc1=float(r[4])) for r in results]),
              open(f'/simurgh2/users/juze/calibjoint/photo_offset_{SEQ}_{OBJ}.json', 'w'))
print('PHOTO_OFFSET_DONE', flush=True)
