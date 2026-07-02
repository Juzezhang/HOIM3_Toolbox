#!/usr/bin/env python3
"""Overlay fitted object 6-DoF poses (objpose_v3) on multi-view HOI-M3 images.

Verifies the /simurgh2/datasets/HOI-M3/objpose_v3 object-pose fits by loading each
object's scanned mesh, placing it at its fitted (object_R, object_T) in the shared
ground/world frame, and rendering ALL objects of a sequence together onto ONE real
camera view (still PNG) -- and optionally animating over a frame range (GIF/MP4).

Run in the `hodome` conda env (has trimesh + pyrender):
    PYOPENGL_PLATFORM=egl /simurgh2/users/juze/anaconda3/envs/hodome/bin/python \
        scripts/visualize_objpose.py \
        --seq bedroom_data01 --frame 0 --view 7 \
        --out /simurgh2/users/juze/hodome_benchmark_release/objpose_vis/bedroom_data01.png

Coordinate handling (identical proven path to visualize_smplx_grid.py):
    * pose json gives object_R (3x3) and object_T (3,) in the ground/world frame.
    * mesh verts V_mesh -> world:
        convention 'R'  : V_world = (object_R   @ V_mesh.T).T + object_T
        convention 'Rt' : V_world = (object_R.T @ V_mesh.T).T + object_T
      (the dataset once stored rotations transposed -- so we try both.)
    * We hand the mvbodyfit pyrender Renderer the world verts + the REAL per-view
      extrinsics cam={K_scaled, R, T}. The wrapper does world->cam
      (X_cam = X_world @ R.T + T.T) and the 180deg-about-X OpenGL flip internally --
      NO manual vertex flip here.

K rescale: refined calibration K is at the ORIGINAL capture resolution (4K for
bedroom/diningroom, 720p for livingroom/office) while images on disk are 720p. We
scale fx,fy,cx,cy by (image_height / calib_H).

Object pose json frames are stored contiguously from 000000; frame 0 == image
frame 0 (start_frame is 0 for all fitted objects), so a still at --frame 0 is a
clean apples-to-apples overlay.
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
import trimesh
from tqdm import tqdm

sys.path.insert(0, '/simurgh/u/juze/code/mv-bodyfit')

from mvbodyfit.core.visualize.pyrender_wrapper import Renderer  # noqa: E402
from mvbodyfit.core.mytools.camera_utils import read_cameras_refined_json  # noqa: E402

OBJPOSE_ROOT = '/simurgh2/datasets/HOI-M3/objpose_v3'
MOCAP_GROUND_ROOT = '/simurgh/group/juze/datasets/HOI-M3/mocap_ground'
SCANNED_ROOT = '/simurgh/group/juze/datasets/HOI-M3/scanned_object'
CALIB_ROOT = '/simurgh2/datasets/HOI-M3/calib_ground_refined'
IMG_ROOT = '/simurgh2/datasets/HOI-M3/images'
DATASET_INFO = '/simurgh2/datasets/HOI-M3/dataset_information.json'


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


def resolve_mesh_path(obj_name, meta):
    """meta may or may not carry an explicit mesh_path; fall back to the standard
    scanned_object/{obj}/{obj}_{variant}.obj layout."""
    mp = meta.get('mesh_path')
    if mp and os.path.exists(mp):
        return mp
    variant = meta.get('mesh_variant', 'simplified_transformed')
    cand = os.path.join(SCANNED_ROOT, obj_name, f'{obj_name}_{variant}.obj')
    if os.path.exists(cand):
        return cand
    # last resort: any obj in the folder
    objs = sorted(Path(os.path.join(SCANNED_ROOT, obj_name)).glob('*.obj'))
    return str(objs[0]) if objs else None


def _load_mesh(name, meta):
    """Resolve + load a scanned mesh. Returns (verts, faces, mesh_path) or None."""
    mesh_path = resolve_mesh_path(name, meta)
    if mesh_path is None:
        return None
    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    return verts, faces, mesh_path


def discover_objects_objpose_v3(seq):
    """Return list of object dicts for every objpose_v3 object subfolder of `seq`
    that has a json/ pose dir and a loadable mesh."""
    seq_dir = os.path.join(OBJPOSE_ROOT, seq)
    objects = []
    for name in sorted(os.listdir(seq_dir)):
        od = os.path.join(seq_dir, name)
        if not os.path.isdir(od):
            continue
        if not os.path.isdir(os.path.join(od, 'json')):
            continue
        meta_path = os.path.join(od, 'meta.json')
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        loaded = _load_mesh(name, meta)
        if loaded is None:
            print(f'[viz]   object {name}: NO mesh found, skipping')
            continue
        verts, faces, mesh_path = loaded
        objects.append({'name': name, 'source': 'objpose_v3', 'dir': od,
                        'meta': meta, 'mesh_path': mesh_path,
                        'verts': verts, 'faces': faces})
    return objects


def discover_objects_mocap_ground(seq):
    """Return list of object dicts read from the RAW GT mocap_ground npz.

    `{seq}_object.npz` key `object_params` is an (n_frames,) object array; each
    element is a dict {obj_name: {'object_R': (3,3), 'object_T': (1,3)}}. We take
    the object roster from frame 0, load each scanned mesh, and stash the whole
    object_params array on every object dict so load_pose can index it per frame.
    Objects without a scanned mesh are skipped and logged."""
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


def discover_objects(seq, source='objpose_v3'):
    if source == 'mocap_ground':
        return discover_objects_mocap_ground(seq)
    return discover_objects_objpose_v3(seq)


def load_pose(ob, frame):
    """Return (R, T) in the world/ground frame for object `ob` at `frame`, or None
    if the object is absent at that frame. Works for both pose sources."""
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


def pose_to_world(verts, R, T, convention):
    """verts (N,3) mesh-frame -> world-frame."""
    if convention == 'R':
        return verts @ R.T + T           # (R @ V.T).T + T
    elif convention == 'Rt':
        return verts @ R + T             # (R.T @ V.T).T + T
    raise ValueError(convention)


def build_render_data(objects, frame, convention):
    """Return {vid: {vertices, faces, vid, name}} for objects present at `frame`."""
    render_data = {}
    present = []
    for vid, ob in enumerate(objects):
        pose = load_pose(ob, frame)
        if pose is None:
            continue
        R, T = pose
        vw = pose_to_world(ob['verts'], R, T, convention)
        render_data[vid] = {'vertices': vw, 'faces': ob['faces'],
                            'vid': vid, 'name': ob['name'], 'smooth': False}
        present.append(ob['name'])
    return render_data, present


def render_view(renderer, render_data, img, cam):
    Ks, _ = scale_K(cam['K'], img.shape[0], cam['H'])
    cam_render = {'K': Ks, 'R': cam['R'], 'T': cam['T'].reshape(3, 1)}
    if not render_data:
        return img
    try:
        return renderer.render_image(render_data, img, cam_render, [])[2][0]
    except Exception as e:  # noqa: BLE001
        print(f'[viz]   render failed: {e}')
        return img


def legend(img, objects, convention):
    """Draw a small color legend (matches get_colors int palette, BGR)."""
    from mvbodyfit.core.visualize.pyrender_wrapper import get_colors
    y = 60
    cv2.putText(img, f'conv={convention}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    for vid, ob in enumerate(objects):
        col = get_colors(vid)  # RGB floats in [0,1]; image is BGR so reverse
        bgr = (int(col[2] * 255), int(col[1] * 255), int(col[0] * 255))
        cv2.putText(img, ob['name'], (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, bgr, 2)
        y += 26
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--seq', default='bedroom_data01')
    ap.add_argument('--source', choices=['objpose_v3', 'mocap_ground'],
                    default='objpose_v3',
                    help="pose source: 'objpose_v3' fits (default) or raw GT "
                         "'mocap_ground' the fits were initialized from")
    ap.add_argument('--frame', type=int, default=0)
    ap.add_argument('--view', default='7', help='single camera id')
    ap.add_argument('--out', required=True, help='output .png (still) or .gif/.mp4 (anim)')
    ap.add_argument('--convention', choices=['R', 'Rt'], default='R',
                    help="rotation convention: 'R' = R@V (v3 default), 'Rt' = R.T@V")
    ap.add_argument('--both', action='store_true',
                    help='render both conventions side-by-side (still only)')
    ap.add_argument('--gpu', type=int, default=0)
    # animation options
    ap.add_argument('--anim', action='store_true', help='animate over frame range')
    ap.add_argument('--start_frame', type=int, default=0)
    ap.add_argument('--end_frame', type=int, default=60)
    ap.add_argument('--step', type=int, default=1)
    ap.add_argument('--fps', type=int, default=10)
    ap.add_argument('--width', type=int, default=0)
    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    view = str(args.view)

    seq2date = build_seq2date()
    date = seq2date.get(args.seq)
    if date is None:
        raise SystemExit(f'No date for seq {args.seq} in dataset_information.json')
    cams = read_cameras_refined_json(os.path.join(CALIB_ROOT, date, 'calibration.json'))
    if view not in cams:
        raise SystemExit(f'view {view} has no calibration entry (have {sorted(cams)[:12]}...)')
    cam = cams[view]

    objects = discover_objects(args.seq, args.source)
    if not objects:
        raise SystemExit(f'No objects found for seq {args.seq} (source={args.source})')
    print(f'[viz] seq={args.seq} source={args.source} date={date} view={view} '
          f'objects={[o["name"] for o in objects]}')

    renderer = Renderer()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    if args.anim:
        frames = list(range(args.start_frame, args.end_frame, args.step))
        out_frames = []
        for f in tqdm(frames, desc=f'{args.seq} v{view}'):
            img_path = os.path.join(IMG_ROOT, args.seq, view, f'{f:06d}.jpg')
            img = cv2.imread(img_path)
            if img is None:
                continue
            rd, _ = build_render_data(objects, f, args.convention)
            img = render_view(renderer, rd, img, cam)
            if args.width and img.shape[1] != args.width:
                h = int(round(img.shape[0] * args.width / img.shape[1]))
                img = cv2.resize(img, (args.width, h), interpolation=cv2.INTER_AREA)
            out_frames.append(img)
        if not out_frames:
            raise SystemExit('[viz] no frames rendered')
        h, w = out_frames[0].shape[:2]
        ext = os.path.splitext(args.out)[1].lower()
        if ext == '.mp4':
            wr = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (w, h))
            for im in out_frames:
                wr.write(im)
            wr.release()
        else:
            import imageio
            imageio.mimsave(args.out, [f[..., ::-1] for f in out_frames],
                            format='GIF', fps=args.fps, loop=0)
        print(f'[viz] wrote {args.out}  {w}x{h}  {len(out_frames)} frames')
        return

    # ---- still ----
    img_path = os.path.join(IMG_ROOT, args.seq, view, f'{args.frame:06d}.jpg')
    base = cv2.imread(img_path)
    if base is None:
        raise SystemExit(f'missing image {img_path}')

    def render_conv(conv):
        rd, present = build_render_data(objects, args.frame, conv)
        print(f'[viz]   conv={conv}: objects present at frame {args.frame}: {present}')
        out = render_view(renderer, rd, base.copy(), cam)
        return legend(out, objects, conv)

    if args.both:
        left = render_conv('R')
        right = render_conv('Rt')
        panel = np.hstack([left, right])
        cv2.putText(panel, f'{args.seq} [{args.source}] f{args.frame} v{view}  '
                    f'[LEFT R | RIGHT R.T]',
                    (10, panel.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)
        cv2.imwrite(args.out, panel)
    else:
        out = render_conv(args.convention)
        cv2.putText(out, f'{args.seq} [{args.source}] f{args.frame} v{view} '
                    f'conv={args.convention}',
                    (10, out.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)
        cv2.imwrite(args.out, out)
    print(f'[viz] wrote {args.out}')


if __name__ == '__main__':
    main()
