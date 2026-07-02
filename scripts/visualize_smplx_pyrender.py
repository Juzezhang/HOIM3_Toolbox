#!/usr/bin/env python3
"""Animated single-view overlay of converted SMPL-X meshes (MHR->SMPL-X) on HOI-M3 video.

Mirrors the HoDome toolbox's ``hodome_visualize_pyrender.py`` look: solid-shaded,
color-coded SMPL-X bodies projected onto ONE camera's real frames and animated over
a frame range into an MP4 or GIF. Renders ALL persons of a sequence (person0,
person1, ...) per frame.

Run in the ``hodome`` conda env (has smplx + pyrender):
    PYOPENGL_PLATFORM=egl /simurgh2/users/juze/anaconda3/envs/hodome/bin/python \
        scripts/visualize_smplx_pyrender.py \
        --seq bedroom_data01 --view 7 --start_frame 0 --end_frame 600 --step 6 \
        --out assets/hoim3_smplx_bedroom.gif --width 480 --fps 10

Coordinate handling is identical to ``visualize_smplx_grid.py`` (fully solved there):
the npz params live in the HOI-M3 ground/world frame. We forward SMPL-X to world-frame
vertices and hand the mvbodyfit pyrender ``Renderer`` the world verts + REAL per-view
extrinsics cam={K_scaled, R, T}. The wrapper does world->cam (X_cam = X_world @ R.T + T.T)
and the 180-deg-about-X OpenGL flip internally -- NO manual vertex flip here.

K rescale: refined calibration K is at the ORIGINAL capture resolution (4K for
bedroom/diningroom/fitnessroom, 720p for livingroom/office) while images on disk are
720p. We scale fx,fy,cx,cy by (image_height / calib_H).
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
from tqdm import tqdm

sys.path.insert(0, '/simurgh/u/juze/code/mv-bodyfit')

import smplx  # noqa: E402
from mvbodyfit.core.visualize.pyrender_wrapper import Renderer  # noqa: E402
from mvbodyfit.core.mytools.camera_utils import read_cameras_refined_json  # noqa: E402

SMPLX_MODEL_DIR = '/simurgh2/users/juze/smplx_models'
SMPLX_ROOT = '/simurgh2/datasets/HOI-M3/smplx_from_mhr'
CALIB_ROOT = '/simurgh2/datasets/HOI-M3/calib_ground_refined'
IMG_ROOT = '/simurgh2/datasets/HOI-M3/images'
DATASET_INFO = '/simurgh2/datasets/HOI-M3/dataset_information.json'

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


def scale_K(K, img_h, calib_H):
    s = img_h / float(calib_H)
    Ks = K.copy().astype(np.float32)
    Ks[0, 0] *= s
    Ks[1, 1] *= s
    Ks[0, 2] *= s
    Ks[1, 2] *= s
    return Ks, s


def forward_person_frames(model, npz_path, frames, device, batch=256):
    """Forward SMPL-X for the requested `frames` (world verts).

    Returns dict {frame_id: verts (N,3)} only for frames present in the npz.
    Batches over frames for one person (betas are constant per person).
    """
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

    # gather params for the selected rows
    params = {}
    for key, dim in _PARAM_KEYS.items():
        if key not in d.files:
            continue
        arr = np.asarray(d[key], dtype=np.float32)
        if arr.ndim == 1:  # e.g. constant betas stored as (dim,)
            arr = np.broadcast_to(arr.reshape(1, -1), (len(frame_ids), arr.shape[-1]))
        params[key] = arr[rows][:, :dim]

    verts_out = {}
    with torch.no_grad():
        for s in range(0, len(rows), batch):
            e = min(s + batch, len(rows))
            kwargs = {k: torch.tensor(v[s:e], dtype=torch.float32, device=device)
                      for k, v in params.items()}
            out = model(**kwargs)
            v = out.vertices.cpu().numpy()
            for j in range(e - s):
                verts_out[present[s + j]] = v[j]
    return verts_out


def write_gif(bgr_frames, out_path, fps, max_colors=128):
    """Write an optimized GIF. Prefers ffmpeg two-pass palettegen/paletteuse
    (global palette + delta frames => small files for photographic content);
    falls back to imageio if ffmpeg is unavailable."""
    import shutil
    import subprocess
    import tempfile

    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        import imageio
        imageio.mimsave(out_path, [f[..., ::-1] for f in bgr_frames],
                        format='GIF', fps=fps, loop=0)
        return

    tmp = tempfile.mkdtemp(prefix='hoim3_gif_')
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
    ap.add_argument('--view', default='0', help='single camera id')
    ap.add_argument('--start_frame', type=int, default=0)
    ap.add_argument('--end_frame', type=int, default=600)
    ap.add_argument('--step', type=int, default=6)
    ap.add_argument('--out', required=True, help='output .mp4 or .gif path')
    ap.add_argument('--width', type=int, default=0,
                    help='downscale output to this width (0 = keep source width)')
    ap.add_argument('--fps', type=int, default=10)
    ap.add_argument('--max_colors', type=int, default=128,
                    help='GIF palette size (lower = smaller file)')
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    view = str(args.view)
    ext = os.path.splitext(args.out)[1].lower()
    if ext not in ('.mp4', '.gif'):
        raise SystemExit(f'--out must end in .mp4 or .gif, got {ext}')

    seq2date = build_seq2date()
    date = seq2date.get(args.seq)
    if date is None:
        raise SystemExit(f'No date for seq {args.seq} in dataset_information.json')
    cams = read_cameras_refined_json(os.path.join(CALIB_ROOT, date, 'calibration.json'))
    if view not in cams:
        raise SystemExit(f'view {view} has no calibration entry (have {sorted(cams)[:10]}...)')

    persons = discover_persons(args.seq)
    if not persons:
        raise SystemExit(f'No SMPL-X npz found for seq {args.seq} under {SMPLX_ROOT}')

    frames = list(range(args.start_frame, args.end_frame, args.step))
    print(f'[viz] seq={args.seq} date={date} view={view} device={device} '
          f'frames={len(frames)} ({args.start_frame}..{args.end_frame} step {args.step}) '
          f'persons={[p for p, _ in persons]}')

    model = build_model(device)
    renderer = Renderer()
    faces = model.faces.astype(np.int32)

    # Pre-forward every person over the whole requested frame set (world verts).
    person_verts = {}
    for pid, npz_path in persons:
        person_verts[pid] = forward_person_frames(model, npz_path, frames, device)
        print(f'[viz]   person{pid}: {len(person_verts[pid])}/{len(frames)} frames present')

    cam = cams[view]
    out_frames = []
    for f in tqdm(frames, desc=f'{args.seq} v{view}'):
        img_path = os.path.join(IMG_ROOT, args.seq, view, f'{f:06d}.jpg')
        img = cv2.imread(img_path)
        if img is None:
            continue
        Ks, _ = scale_K(cam['K'], img.shape[0], cam['H'])
        cam_render = {'K': Ks, 'R': cam['R'], 'T': cam['T'].reshape(3, 1)}

        render_data = {}
        for pid, vmap in person_verts.items():
            if f in vmap:
                render_data[pid] = {'vertices': vmap[f], 'faces': faces,
                                    'vid': pid, 'name': f'person_{pid}'}
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
    else:  # .gif
        write_gif(out_frames, args.out, args.fps, args.max_colors)

    size_mb = os.path.getsize(args.out) / 1e6
    print(f'[viz] wrote {args.out}  {w}x{h}  {len(out_frames)} frames  {size_mb:.2f} MB')


if __name__ == '__main__':
    main()
