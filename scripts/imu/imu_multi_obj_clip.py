"""Generalized dense clip: <SEQ>,<OBJ> from env — IMU-R (hand-eye A,B) + fitted-T
vs mocap_ground, overlaid on images around the object's BIGGEST motion event.
Auto: motion window from IMU rel-angles; show-views = 3 largest-mask views.
mocap index offset = IMU json start_frame (proven for data03 via reprojection).
"""
import os, json
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import numpy as np, cv2, trimesh, pyrender
from scipy.optimize import minimize
import sys as _sys
_sys.path.insert(0, '/simurgh/u/juze/code/mv-bodyfit')
_sys.path.insert(0, '/simurgh/u/juze/code/mv-bodyfit/tools')
_cwd0 = os.getcwd(); os.chdir('/simurgh/u/juze/code/mv-bodyfit')
import visualize_results as VR
os.chdir(_cwd0)

SEQ = os.environ['SEQ']; OBJ = os.environ['OBJ']
D = '/simurgh/group/juze/datasets/HOI-M3'
D2 = '/simurgh2/datasets/HOI-M3'
CT = f'{D2}/cutie_tracking/{SEQ}'
OUT = f'/simurgh2/users/juze/calibjoint/imu_clip_{SEQ}_{OBJ}.mp4'
INFO = json.load(open('/simurgh/group/juze/datasets/HOI-M3/dataset_information.json'))
DATE = next(d for d, ss in INFO.items() if SEQ in ss)

scan = json.load(open(f'/simurgh2/users/juze/calibjoint/imu_scan/{SEQ}.json'))
so = scan['objects'][OBJ]
A = np.array(so['A']); B = np.array(so['B']); HE = so['handeye_res_deg']
DIV = so.get('divergence_segments', [])
dj = json.load(open(f'{D}/imu_data/{SEQ}/imu_data/{OBJ}.json'))
Rj = np.array(dj['object_R']); s = int(dj['start_frame'])
mg = np.load(f'{D}/mocap_ground/{SEQ}_object.npz', allow_pickle=True)['object_params']
def mgp(fr):
    e = mg[fr - s][OBJ]; e = e.item() if hasattr(e, 'item') else e
    return np.array(e['object_R']), np.array(e['object_T']).ravel()

names = json.load(open(f'{D2}/cutie_refs/{SEQ}/masks/0_names.json'))['mask_names']
OBJ_IDX = names.index(OBJ) + 1
PERSON_IDX = tuple(i + 1 for i, n in enumerate(names) if n.startswith('person'))

def ang(Ra, Rb):
    return np.degrees(np.arccos(np.clip((np.trace(Ra @ Rb.T) - 1) / 2, -1, 1)))

# ---- auto motion window: max cumulative IMU rotation over 1600 frames ----
st = 40
rel = np.array([ang(Rj[i + st], Rj[i]) for i in range(0, len(Rj) - st, st)])
csum = np.cumsum(rel)
Wn = 1600 // st
best_i = int(np.argmax(csum[Wn:] - csum[:-Wn])) if len(csum) > Wn else 0
f0 = s + best_i * st; f1 = min(f0 + 1600, s + len(Rj) - 1)
FRAMES = list(range(f0, f1, 8))
print(f'{SEQ}/{OBJ}: motion window {f0}-{f1} (cum rot {csum[min(best_i+Wn,len(csum)-1)]-csum[best_i]:.0f} deg), '
      f'handeye {HE:.2f} deg', flush=True)

cal = json.load(open(f'{D2}/calib_ground_refined/{DATE}/calibration.json'))
ALLV = [0, 2, 5, 7, 10, 14, 19, 21, 24, 29, 36]
cams = {}
for v in ALLV:
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
        if r.sum() < 150: continue
        out[v] = (r, np.isin(m, PERSON_IDX))
    if len(out) > cap:
        keep = sorted(out, key=lambda v: -out[v][0].sum())[:cap]
        out = {v: out[v] for v in keep}
    return out

# show-views: 3 with biggest average object mask over 12 probe frames
probe = {v: 0 for v in cams}
for fr in FRAMES[::max(1, len(FRAMES)//12)]:
    for v, (r, _) in masks_at(fr, cap=99).items():
        probe[v] += r.sum()
SHOW = [v for v, _ in sorted(probe.items(), key=lambda x: -x[1])[:3]]
print('show views:', SHOW, flush=True)

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

def overlay(img, S, col):
    o = img.copy()
    o[S] = (0.55 * o[S] + 0.45 * np.array(col)).astype(np.uint8)
    cnts, _ = cv2.findContours(S.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(o, cnts, -1, col, 2)
    return o

w = None; prev_T = None; means = {'mocap': [], 'imu': []}
for fr in FRAMES:
    if fr - s >= len(Rj) or fr - s >= len(mg): break
    msk = masks_at(fr)
    Rimu = A @ Rj[fr - s] @ B
    Rm, Tm = mgp(fr)
    carried = False
    if len(msk) >= 2:
        T0 = tri_T(msk)
        if prev_T is not None and occ_iou(Rimu, prev_T, msk) > occ_iou(Rimu, T0, msk):
            T0 = prev_T
        r_ = minimize(lambda T: -occ_iou(Rimu, T, msk), T0, method='Nelder-Mead',
                      options={'maxfev': 35, 'xatol': 3e-3, 'fatol': 1e-3})
        Tf, iou_imu = r_.x, -r_.fun; prev_T = Tf
    elif prev_T is not None:
        Tf = prev_T; carried = True; iou_imu = occ_iou(Rimu, Tf, msk) if msk else 0.0
    else:
        continue
    iou_mg = occ_iou(Rm, Tm, msk) if msk else 0.0
    if msk:
        means['mocap'].append(iou_mg); means['imu'].append(iou_imu)
    rows = []
    for label, R_, T_, col, iou in [
            ('mocap_ground (released)', Rm, Tm, (60, 60, 230), iou_mg),
            ('IMU-R (hand-eye) + fitted T', Rimu, Tf, (60, 200, 60), iou_imu)]:
        tiles = []
        for v in SHOW:
            img = cv2.imread(f'{D2}/images/{SEQ}/{v}/{fr:06d}.jpg')
            if img is None: img = np.zeros((720, 1280, 3), np.uint8)
            wv = (R_ @ np.asarray(tm.vertices).T).T + T_   # object verts -> world
            K_, Rc_, tc_ = cams[v]
            t_ = VR.render_mesh_pyrender(img, wv, np.asarray(tm.faces), K_, Rc_, tc_.reshape(3, 1),
                                         pid=(0 if 'mocap' in label else 1))
            cv2.putText(t_, f'v{v} f{fr}', (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            tiles.append(cv2.resize(t_, (520, 293)))
        row = np.hstack(tiles)
        tag = ' [DIVERGENCE]' if any(a <= fr <= b for a, b in DIV) else ''
        cv2.putText(row, f'{label} IoU={iou:.3f}{tag}{" (T carried)" if carried else ""}',
                    (12, row.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2)
        rows.append(row)
    canvas = np.vstack(rows)
    cv2.putText(canvas, f'{OBJ} @ {SEQ} | handeye {HE:.2f}deg | top=mocap bottom=IMU-R+fitT', (12, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    if w is None:
        w = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*'mp4v'), 7, (canvas.shape[1], canvas.shape[0]))
    w.write(canvas)
if w: w.release()
print(f"means: mocap={np.mean(means['mocap']):.3f} imu={np.mean(means['imu']):.3f} n={len(means['imu'])}", flush=True)
print('WROTE', OUT, flush=True)
