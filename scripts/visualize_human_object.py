#!/usr/bin/env python3
"""Animated single-view overlay of SMPL-X humans AND scanned object meshes on HOI-M3 video.

Combines the two proven single-modality visualizers of this repo into ONE pyrender
pass per frame:
  * humans  -- ``visualize_smplx_pyrender.py``: forwards each ``smplx/{seq}_person{id}.npz``
    (MHR->SMPL-X, HOI-M3 world/ground frame) to world-frame vertices.
  * objects -- ``visualize_objpose.py``: poses each scanned mesh by (object_R, object_T)
    into the same world frame, from ``--obj_source {objpose_v3, mocap_ground}``.

Both feed world-frame verts + the REAL per-view extrinsics cam={K_scaled, R, T} read from
``calib_ground_refined/{date}/calibration.json`` (the SAME cameras MHR used) to the
mvbodyfit pyrender ``Renderer``, which does world->cam and the 180-deg-about-X OpenGL flip
internally -- NO manual vertex flip here. Rotation convention for objects is plain ``R``
(V_world = V_mesh @ R.T + T). K is rescaled by (image_height / calib_H).

Humans are drawn in warm/skin tones; objects in distinct cool colors (a per-mesh RGB tuple
is passed as ``vid`` -- the renderer's ``get_colors`` returns a 3-tuple verbatim as the
material baseColorFactor). Frames iterate the intersection of available human frames
(npz ``frame_ids``) and object frames, restricted to [start_frame, end_frame) and
subsampled by ``step``.

Run in the ``hodome`` conda env (has smplx + trimesh + pyrender):
    PYOPENGL_PLATFORM=egl /simurgh2/users/juze/anaconda3/envs/hodome/bin/python \
        scripts/visualize_human_object.py \
        --seq bedroom_data01 --view 7 --obj_source objpose_v3 \
        --start_frame 0 --end_frame 600 --step 10 \
        --out assets/hoim3_ho_bedroom.gif --width 480 --fps 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import cv2
import numpy as np
import torch
import trimesh
from tqdm import tqdm

sys.path.insert(0, '/simurgh/u/juze/code/mv-bodyfit')

import smplx  # noqa: E402
from mvbodyfit.core.visualize.pyrender_wrapper import Renderer  # noqa: E402
from mvbodyfit.core.mytools.camera_utils import read_cameras_refined_json  # noqa: E402

SMPLX_MODEL_DIR = '/simurgh2/users/juze/smplx_models'
SMPLX_ROOT = '/simurgh2/datasets/HOI-M3/smplx'
OBJPOSE_ROOT = '/simurgh2/datasets/HOI-M3/objpose_v3'
MOCAP_GROUND_ROOT = '/simurgh/group/juze/datasets/HOI-M3/mocap_ground'
SCANNED_ROOT = '/simurgh/group/juze/datasets/HOI-M3/scanned_object'
IMG_ROOT = '/simurgh2/datasets/HOI-M3/images'
DATASET_INFO = '/simurgh2/datasets/HOI-M3/dataset_information.json'

# Two calibrations ship with the dataset (see README). They live in DIFFERENT world
# frames and are HARD-PAIRED with their own fits -- never mix them:
#   ground_refined  : refined PINHOLE calib (distCoeff ~= 0). Pairs with smplx/ + mhr/.
#   with_distortion : the MORE ACCURATE calib (5-param OpenCV distortion, genuinely
#                     non-zero). Fits were triangulated from UNDISTORTED 2D, so the
#                     image MUST be undistorted before overlay. Pairs with
#                     smplx_with_distortion/ + mhr_withdist/.
CALIB_ROOTS = {
    'ground_refined': '/simurgh2/datasets/HOI-M3/calib_ground_refined',
    'with_distortion': '/simurgh2/datasets/HOI-M3/calib_with_distortion',
}

# Warm / skin tones for humans (RGB floats). Cycled per person.
HUMAN_COLORS = [
    (0.96, 0.76, 0.62),   # light skin
    (0.90, 0.60, 0.48),   # terracotta / warm tan
    (0.98, 0.84, 0.55),   # warm sand
    (0.85, 0.52, 0.42),   # clay
]
# Distinct cool colors for objects (RGB floats). Cycled per object.
OBJECT_COLORS = [
    (0.36, 0.49, 0.90),   # blue
    (0.25, 0.72, 0.68),   # teal
    (0.55, 0.47, 0.82),   # purple
    (0.30, 0.66, 0.92),   # sky
    (0.35, 0.80, 0.55),   # green
    (0.20, 0.52, 0.72),   # steel
]

# axis-angle / pose keys forwarded to SMPL-X
_PARAM_KEYS = {
    'betas': 10, 'global_orient': 3, 'body_pose': 63,
    'left_hand_pose': 45, 'right_hand_pose': 45,
    'jaw_pose': 3, 'leye_pose': 3, 'reye_pose': 3,
    'expression': 10, 'transl': 3,
}


def build_seq2date():
    info = json.load(open(DATASET_INFO))
    seq2date = {}
    for date, entries in info.items():
        for e in entries:
            if isinstance(e, str):
                seq2date[e] = date
    return seq2date


def scale_K(K, img_h, calib_H):
    s = img_h / float(calib_H)
    Ks = K.copy().astype(np.float32)
    Ks[0, 0] *= s
    Ks[1, 1] *= s
    Ks[0, 2] *= s
    Ks[1, 2] *= s
    return Ks, s


def maybe_undistort(img, Ks, dist, use_distortion):
    """Undistort `img` in place of the raw frame when using the distortion calib.

    The with_distortion fits were triangulated from UNDISTORTED 2D, so we rectify the
    image with cv2.undistort using the resolution-scaled K and the RAW distCoeff, then
    project the mesh with the SAME Ks. cv2.undistort defaults newCameraMatrix=Ks, so the
    rectified image matches Ks exactly. distCoeff is dimensionless (normalized coords),
    so it does NOT scale with resolution -- pass it raw alongside the scaled K.
    For ground_refined (distCoeff ~= 0) this is a no-op and we skip it entirely.
    """
    if not use_distortion:
        return img
    dist = np.asarray(dist, dtype=np.float32).reshape(-1)
    if dist.size == 0 or not np.any(dist != 0):
        return img
    return cv2.undistort(img, Ks, dist)


# ---------------------------------------------------------------------------
# Humans (SMPL-X)  -- mirrors visualize_smplx_pyrender.py
# ---------------------------------------------------------------------------
def build_model(device, num_betas=10):
    model = smplx.create(
        SMPLX_MODEL_DIR, model_type='smplx', gender='neutral',
        use_pca=False, flat_hand_mean=True,
        num_betas=num_betas, num_expression_coeffs=10,
    ).to(device)
    model.eval()
    return model


_MODEL_CACHE = {}


def model_for_betas(device, num_betas):
    """smplx_with_distortion is MIXED provenance: MHR-sourced files carry 10
    betas, DenseFit-sourced carry 16 (meta `source` field). Cache one model per
    beta dim so both render correctly (never truncate 16->10 silently)."""
    key = (str(device), int(num_betas))
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = build_model(device, num_betas=int(num_betas))
    return _MODEL_CACHE[key]


def discover_persons(seq):
    persons = []
    for p in sorted(Path(SMPLX_ROOT).glob(f'{seq}_person*.npz')):
        pid = int(p.stem.split('person')[-1])
        persons.append((pid, str(p)))
    return persons


def person_frame_ids(npz_path):
    return set(int(f) for f in np.load(npz_path)['frame_ids'])


def forward_person_frames(model, npz_path, frames, device, batch=256):
    """Forward SMPL-X for requested `frames` -> {frame_id: verts (N,3)} (world frame)."""
    d = np.load(npz_path)
    frame_ids = d['frame_ids']
    fid2row = {int(f): i for i, f in enumerate(frame_ids)}
    rows, present = [], []
    for f in frames:
        if int(f) in fid2row:
            rows.append(fid2row[int(f)])
            present.append(int(f))
    if not rows:
        return {}
    rows = np.asarray(rows, dtype=np.int64)

    params = {}
    for key, dim in _PARAM_KEYS.items():
        if key not in d.files:
            continue
        arr = np.asarray(d[key], dtype=np.float32)
        if arr.ndim == 1:
            arr = np.broadcast_to(arr.reshape(1, -1), (len(frame_ids), arr.shape[-1]))
        params[key] = arr[rows][:, :dim]

    verts_out = {}
    with torch.no_grad():
        for s in range(0, len(rows), batch):
            e = min(s + batch, len(rows))
            kwargs = {k: torch.tensor(v[s:e], dtype=torch.float32, device=device)
                      for k, v in params.items()}
            nb = params['betas'].shape[-1] if 'betas' in params else 10
            out = model_for_betas(device, nb)(**kwargs)
            v = out.vertices.cpu().numpy()
            for j in range(e - s):
                verts_out[present[s + j]] = v[j]
    return verts_out


# ---------------------------------------------------------------------------
# Objects (scanned meshes)  -- mirrors visualize_objpose.py
# ---------------------------------------------------------------------------
def resolve_mesh_path(obj_name, meta):
    mp = meta.get('mesh_path')
    if mp and os.path.exists(mp):
        return mp
    variant = meta.get('mesh_variant', 'simplified_transformed')
    cand = os.path.join(SCANNED_ROOT, obj_name, f'{obj_name}_{variant}.obj')
    if os.path.exists(cand):
        return cand
    objs = sorted(Path(os.path.join(SCANNED_ROOT, obj_name)).glob('*.obj'))
    return str(objs[0]) if objs else None


def _load_mesh(name, meta):
    mesh_path = resolve_mesh_path(name, meta)
    if mesh_path is None:
        return None
    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    return verts, faces, mesh_path


def discover_objects_objpose_v3(seq):
    seq_dir = os.path.join(OBJPOSE_ROOT, seq)
    objects = []
    for name in sorted(os.listdir(seq_dir)):
        od = os.path.join(seq_dir, name)
        if not os.path.isdir(od) or not os.path.isdir(os.path.join(od, 'json')):
            continue
        meta_path = os.path.join(od, 'meta.json')
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        loaded = _load_mesh(name, meta)
        if loaded is None:
            print(f'[viz]   object {name}: NO scanned mesh, skipping')
            continue
        verts, faces, mesh_path = loaded
        objects.append({'name': name, 'source': 'objpose_v3', 'dir': od,
                        'meta': meta, 'mesh_path': mesh_path,
                        'verts': verts, 'faces': faces})
    return objects


def discover_objects_mocap_ground(seq):
    npz_path = os.path.join(MOCAP_GROUND_ROOT, f'{seq}_object.npz')
    if not os.path.exists(npz_path):
        raise SystemExit(f'mocap_ground: no object npz at {npz_path}')
    object_params = np.load(npz_path, allow_pickle=True)['object_params']
    if len(object_params) == 0 or not isinstance(object_params[0], dict):
        raise SystemExit(f'mocap_ground: unexpected object_params in {npz_path}')
    names = sorted(object_params[0].keys())
    objects = []
    for name in names:
        loaded = _load_mesh(name, {})
        if loaded is None:
            print(f'[viz]   object {name}: NO scanned mesh, skipping')
            continue
        verts, faces, mesh_path = loaded
        objects.append({'name': name, 'source': 'mocap_ground',
                        'object_params': object_params, 'meta': {},
                        'mesh_path': mesh_path, 'verts': verts, 'faces': faces})
    return objects


def discover_objects(seq, source):
    if source == 'mocap_ground':
        return discover_objects_mocap_ground(seq)
    return discover_objects_objpose_v3(seq)


def load_pose(ob, frame):
    """Return (R, T) in the world frame for object `ob` at `frame`, or None if absent."""
    if ob['source'] == 'mocap_ground':
        op = ob['object_params']
        if frame < 0 or frame >= len(op):
            return None
        fd = op[frame]
        if not isinstance(fd, dict) or ob['name'] not in fd:
            return None
        entry = fd[ob['name']]
        R = np.asarray(entry['object_R'], dtype=np.float32).reshape(3, 3)
        T = np.asarray(entry['object_T'], dtype=np.float32).reshape(3)
        return R, T
    jp = os.path.join(ob['dir'], 'json', f'{frame:06d}.json')
    if not os.path.exists(jp):
        return None
    d = json.load(open(jp))
    R = np.asarray(d['object_R'], dtype=np.float32).reshape(3, 3)
    T = np.asarray(d['object_T'], dtype=np.float32).reshape(3)
    return R, T


def object_frame_ids(objects, hint_frames):
    """Union of frames where ANY object is present (world pose available).

    objpose_v3: enumerate the json/ dir per object (contiguous 000000..). mocap_ground:
    frames [0, len(object_params)) where the object appears. `hint_frames` bounds the
    probe range so we don't stat 20k json files needlessly."""
    fs = set()
    for ob in objects:
        if ob['source'] == 'mocap_ground':
            op = ob['object_params']
            for f in hint_frames:
                if 0 <= f < len(op) and isinstance(op[f], dict) and ob['name'] in op[f]:
                    fs.add(f)
        else:
            jdir = os.path.join(ob['dir'], 'json')
            have = set(int(p.stem) for p in Path(jdir).glob('*.json'))
            fs |= (have & set(hint_frames))
    return fs


def pose_to_world(verts, R, T):
    """Plain 'R' convention (confirmed): V_world = (R @ V_mesh.T).T + T."""
    return verts @ R.T + T


# ---------------------------------------------------------------------------
# GIF writer (ffmpeg two-pass palettegen -> small files; imageio fallback)
# ---------------------------------------------------------------------------
def write_gif(bgr_frames, out_path, fps, max_colors=128):
    import shutil
    import subprocess
    import tempfile

    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        import imageio
        imageio.mimsave(out_path, [f[..., ::-1] for f in bgr_frames],
                        format='GIF', fps=fps, loop=0)
        return

    tmp = tempfile.mkdtemp(prefix='hoim3_ho_gif_')
    try:
        for i, im in enumerate(bgr_frames):
            cv2.imwrite(os.path.join(tmp, f'{i:04d}.png'), im)
        pal = os.path.join(tmp, 'palette.png')
        pattern = os.path.join(tmp, '%04d.png')
        subprocess.run(
            [ffmpeg, '-y', '-loglevel', 'error', '-framerate', str(fps), '-i', pattern,
             '-vf', f'palettegen=max_colors={max_colors}:stats_mode=diff', pal],
            check=True)
        subprocess.run(
            [ffmpeg, '-y', '-loglevel', 'error', '-framerate', str(fps), '-i', pattern,
             '-i', pal, '-lavfi',
             'paletteuse=dither=none:diff_mode=rectangle',
             '-loop', '0', out_path],
            check=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--seq', default='bedroom_data01')
    ap.add_argument('--view', default='7', help='single camera id')
    ap.add_argument('--obj_source', choices=['objpose_v3', 'mocap_ground'],
                    default='objpose_v3')
    ap.add_argument('--start_frame', type=int, default=0)
    ap.add_argument('--end_frame', type=int, default=600)
    ap.add_argument('--step', type=int, default=10)
    ap.add_argument('--out', required=True, help='output .mp4 or .gif path')
    ap.add_argument('--width', type=int, default=0,
                    help='downscale output to this width (0 = keep source width)')
    ap.add_argument('--fps', type=int, default=10)
    ap.add_argument('--max_colors', type=int, default=128,
                    help='GIF palette size (lower = smaller file)')
    ap.add_argument('--calib', choices=['ground_refined', 'with_distortion'],
                    default='ground_refined',
                    help="calibration set: 'ground_refined' (pinhole, default) or "
                         "'with_distortion' (more accurate; undistorts the image)")
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    view = str(args.view)
    ext = os.path.splitext(args.out)[1].lower()
    if ext not in ('.mp4', '.gif'):
        raise SystemExit(f'--out must end in .mp4 or .gif, got {ext}')

    use_distortion = args.calib == 'with_distortion'
    calib_root = CALIB_ROOTS[args.calib]
    obj_root = MOCAP_GROUND_ROOT if args.obj_source == 'mocap_ground' else OBJPOSE_ROOT
    if use_distortion and 'with_distortion' not in SMPLX_ROOT:
        print(f'[viz] WARNING: --calib with_distortion but SMPLX_ROOT={SMPLX_ROOT} '
              f'and objects from {obj_root} are the non-distortion fit sets -- these '
              f'live in a different world frame (pair with smplx_with_distortion/ + '
              f'mhr_withdist/).')

    seq2date = build_seq2date()
    date = seq2date.get(args.seq)
    if date is None:
        raise SystemExit(f'No date for seq {args.seq} in dataset_information.json')
    cams = read_cameras_refined_json(os.path.join(calib_root, date, 'calibration.json'))
    if view not in cams:
        raise SystemExit(f'view {view} has no calibration entry (have {sorted(cams)[:12]}...)')
    cam = cams[view]

    persons = discover_persons(args.seq)
    if not persons:
        raise SystemExit(f'No SMPL-X npz found for seq {args.seq} under {SMPLX_ROOT}')
    objects = discover_objects(args.seq, args.obj_source)
    if not objects:
        raise SystemExit(f'No objects found for seq {args.seq} (source={args.obj_source})')

    # --- Frame set: intersect available human frames & object frames, then window+step ---
    hint = list(range(args.start_frame, args.end_frame))
    hframes = set()
    for _, npz in persons:
        hframes |= person_frame_ids(npz)
    oframes = object_frame_ids(objects, hint)
    candidate = sorted(hframes & oframes & set(hint))
    frames = candidate[::args.step]
    if not frames:
        raise SystemExit('[viz] no overlapping human+object frames in the requested window')

    print(f'[viz] seq={args.seq} date={date} view={view} device={device} '
          f'obj_source={args.obj_source}')
    print(f'[viz] persons={[p for p, _ in persons]}  objects={[o["name"] for o in objects]}')
    print(f'[viz] frames={len(frames)}  window=[{args.start_frame},{args.end_frame}) '
          f'step={args.step}  (human&object overlap={len(candidate)})')

    model = build_model(device)
    renderer = Renderer()
    faces = model.faces.astype(np.int32)

    # Pre-forward humans over the selected frames (world verts).
    person_verts = {}
    for pid, npz_path in persons:
        person_verts[pid] = forward_person_frames(model, npz_path, frames, device)
        print(f'[viz]   person{pid}: {len(person_verts[pid])}/{len(frames)} frames present')

    out_frames = []
    for f in tqdm(frames, desc=f'{args.seq} v{view}'):
        img_path = os.path.join(IMG_ROOT, args.seq, view, f'{f:06d}.jpg')
        img = cv2.imread(img_path)
        if img is None:
            continue
        Ks, _ = scale_K(cam['K'], img.shape[0], cam['H'])
        img = maybe_undistort(img, Ks, cam['dist'], use_distortion)
        cam_render = {'K': Ks, 'R': cam['R'], 'T': cam['T'].reshape(3, 1)}

        render_data = {}
        # humans: warm/skin tones, smooth shading
        for i, (pid, vmap) in enumerate(sorted(person_verts.items())):
            if f in vmap:
                col = HUMAN_COLORS[i % len(HUMAN_COLORS)]
                key = f'person_{pid}'
                render_data[key] = {'vertices': vmap[f], 'faces': faces,
                                    'vid': col, 'name': key, 'smooth': True}
        # objects: distinct cool colors, flat (scanned) shading
        for i, ob in enumerate(objects):
            pose = load_pose(ob, f)
            if pose is None:
                continue
            R, T = pose
            vw = pose_to_world(ob['verts'], R, T)
            col = OBJECT_COLORS[i % len(OBJECT_COLORS)]
            key = f'obj_{i}_{ob["name"]}'
            render_data[key] = {'vertices': vw, 'faces': ob['faces'],
                                'vid': col, 'name': key, 'smooth': False}

        if render_data:
            try:
                img = renderer.render_image(render_data, img, cam_render, [])[2][0]
            except Exception as e:  # noqa: BLE001
                print(f'[viz]   frame {f}: render failed: {e}')

        if args.width and img.shape[1] != args.width:
            h = int(round(img.shape[0] * args.width / img.shape[1]))
            img = cv2.resize(img, (args.width, h), interpolation=cv2.INTER_AREA)
        out_frames.append(img)

    if not out_frames:
        raise SystemExit('[viz] no frames rendered (missing images?)')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    h, w = out_frames[0].shape[:2]
    if ext == '.mp4':
        writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*'mp4v'),
                                 args.fps, (w, h))
        for im in out_frames:
            writer.write(im)
        writer.release()
    else:
        write_gif(out_frames, args.out, args.fps, args.max_colors)

    size_mb = os.path.getsize(args.out) / 1e6
    print(f'[viz] wrote {args.out}  {w}x{h}  {len(out_frames)} frames  {size_mb:.2f} MB')


if __name__ == '__main__':
    main()
