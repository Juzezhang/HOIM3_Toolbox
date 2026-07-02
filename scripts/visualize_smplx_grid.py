#!/usr/bin/env python3
"""Overlay converted SMPL-X meshes (MHR->SMPL-X) on multi-view HOI-M3 images.

Verifies the /simurgh2/datasets/HOI-M3/smplx_from_mhr conversions by rendering
the world-frame SMPL-X meshes into each camera using the refined ground
calibration, then tiling the views into a labeled grid PNG.

Run in the `hodome` conda env:
    /simurgh2/users/juze/anaconda3/envs/hodome/bin/python tools/viz_hoim3_smplx.py \
        --seq bedroom_data01 --frame 0 --views 0 7 14 21 28 35 \
        --out /simurgh2/users/juze/hodome_benchmark_release/hoim3_vis/bedroom_data01.png

Coordinate handling: the npz params are in the HOI-M3 ground/world frame. We
forward SMPL-X to world-frame vertices, then hand the mvbodyfit pyrender
Renderer the world verts together with the REAL per-view extrinsics
cam={K_scaled, R, T}. The wrapper computes X_cam = X_world @ R.T + T.T (OpenCV
world->camera) and internally applies the 180-deg-about-X flip to convert to
pyrender's OpenGL camera convention -- so NO manual vertex flip is needed here.

K rescale: the refined calibration K is at the ORIGINAL capture resolution
(4K for bedroom/diningroom/fitnessroom, 720p for livingroom/office) while the
images on disk are 720p. We scale fx,fy,cx,cy by (image_height / calib_H).
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

sys.path.insert(0, '/simurgh/u/juze/code/mv-bodyfit')

import smplx  # noqa: E402
from mvbodyfit.core.visualize.pyrender_wrapper import Renderer  # noqa: E402
from mvbodyfit.core.mytools.camera_utils import read_cameras_refined_json  # noqa: E402

SMPLX_MODEL_DIR = '/simurgh2/users/juze/smplx_models'
SMPLX_ROOT = '/simurgh2/datasets/HOI-M3/smplx_from_mhr'
CALIB_ROOT = '/simurgh2/datasets/HOI-M3/calib_ground_refined'
IMG_ROOT = '/simurgh2/datasets/HOI-M3/images'
DATASET_INFO = '/simurgh2/datasets/HOI-M3/dataset_information.json'

# axis-angle / pose keys forwarded to SMPL-X for a single frame
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


def build_model(device):
    model = smplx.create(
        SMPLX_MODEL_DIR, model_type='smplx', gender='neutral',
        use_pca=False, flat_hand_mean=True,
        num_betas=10, num_expression_coeffs=10,
    ).to(device)
    model.eval()
    return model


def discover_persons(seq):
    persons = []
    for p in sorted(Path(SMPLX_ROOT).glob(f'{seq}_person*.npz')):
        pid = int(p.stem.split('person')[-1])
        persons.append((pid, str(p)))
    return persons


def forward_person(model, npz_path, frame_id, device):
    """Return world-frame vertices (N,3) for the row matching frame_id, or None."""
    d = np.load(npz_path)
    frame_ids = d['frame_ids']
    rows = np.where(frame_ids == frame_id)[0]
    if len(rows) == 0:
        return None
    row = int(rows[0])
    kwargs = {}
    for key, dim in _PARAM_KEYS.items():
        if key not in d.files:
            continue
        arr = np.asarray(d[key][row], dtype=np.float32).reshape(-1)[:dim]
        kwargs[key] = torch.tensor(arr[None], dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model(**kwargs)
    return out.vertices[0].cpu().numpy()


def make_grid(view_imgs, n_cols=3, target_w=640):
    h, w = view_imgs[0].shape[:2]
    target_h = int(h * target_w / w)
    resized = [cv2.resize(im, (target_w, target_h)) for im in view_imgs]
    n_rows = (len(resized) + n_cols - 1) // n_cols
    grid = np.zeros((n_rows * target_h, n_cols * target_w, 3), dtype=np.uint8)
    for i, im in enumerate(resized):
        r, c = i // n_cols, i % n_cols
        grid[r * target_h:(r + 1) * target_h, c * target_w:(c + 1) * target_w] = im
    return grid


def scale_K(K, img_h, calib_H):
    s = img_h / float(calib_H)
    Ks = K.copy().astype(np.float32)
    Ks[0, 0] *= s
    Ks[1, 1] *= s
    Ks[0, 2] *= s
    Ks[1, 2] *= s
    return Ks, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', default='bedroom_data01')
    ap.add_argument('--frame', type=int, default=0)
    ap.add_argument('--views', nargs='+', default=['0', '7', '14', '21', '28', '35'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--n_cols', type=int, default=3)
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    views = [str(v) for v in args.views]

    seq2date = build_seq2date()
    date = seq2date.get(args.seq)
    if date is None:
        raise SystemExit(f'No date for seq {args.seq} in dataset_information.json')
    calib_json = os.path.join(CALIB_ROOT, date, 'calibration.json')
    cams = read_cameras_refined_json(calib_json)

    print(f'[viz] seq={args.seq} date={date} frame={args.frame} views={views} device={device}')

    model = build_model(device)
    renderer = Renderer()

    persons = discover_persons(args.seq)
    if not persons:
        raise SystemExit(f'No SMPL-X npz found for seq {args.seq} under {SMPLX_ROOT}')
    faces = model.faces.astype(np.int32)

    # forward all persons once (world-frame verts reused across views)
    person_verts = {}
    for pid, npz_path in persons:
        v = forward_person(model, npz_path, args.frame, device)
        if v is not None:
            person_verts[pid] = v
    print(f'[viz] persons with verts at frame {args.frame}: {sorted(person_verts.keys())}')

    view_imgs = []
    for v in views:
        img_path = os.path.join(IMG_ROOT, args.seq, v, f'{args.frame:06d}.jpg')
        img = cv2.imread(img_path)
        if img is None:
            print(f'[viz]   view {v}: MISSING image {img_path}')
            view_imgs.append(np.zeros((720, 1280, 3), dtype=np.uint8))
            continue
        if v not in cams:
            print(f'[viz]   view {v}: no calibration entry')
            cv2.putText(img, f'v{v} NO-CALIB', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            view_imgs.append(img)
            continue

        cam = cams[v]
        img_h = img.shape[0]
        Ks, s = scale_K(cam['K'], img_h, cam['H'])
        cam_render = {'K': Ks, 'R': cam['R'], 'T': cam['T'].reshape(3, 1)}

        render_data = {}
        for pid, verts in person_verts.items():
            render_data[pid] = {
                'vertices': verts, 'faces': faces,
                'vid': pid, 'name': f'person_{pid}',
            }
        if render_data:
            try:
                out = renderer.render_image(render_data, img, cam_render, [])
                img = out[2][0]
            except Exception as e:  # noqa: BLE001
                print(f'[viz]   view {v}: render failed: {e}')

        cv2.putText(img, f'v{v}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 255), 3)
        view_imgs.append(img)

    grid = make_grid(view_imgs, n_cols=args.n_cols)
    label = (f'{args.seq} | frame {args.frame} | date {date} | '
             f'persons={sorted(person_verts.keys())} | views={views}')
    cv2.putText(grid, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cv2.imwrite(args.out, grid)
    print(f'[viz] wrote {args.out}  ({grid.shape[1]}x{grid.shape[0]})')


if __name__ == '__main__':
    main()
