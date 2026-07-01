#!/usr/bin/env python3
"""Reproject fitted 3D keypoints to all 42 camera views, build 6x7 grid mp4 at 10 fps.

Input :
  - /simurgh2/datasets/HOI-M3/mhr_simplified/<seq>/keypoints3d/<fid:06d>.json
  - /simurgh2/datasets/HOI-M3/calib_ground_refined/<date>/calibration.json
  - /simurgh/group/juze/datasets/HOI-M3/dataset_information.json
  - /simurgh2/datasets/HOI-M3/seq_unavail_views.json
  - /simurgh2/datasets/HOI-M3/images/<seq>/<view>/<fid:06d>.jpg  (1280x720)

Output:
  - /simurgh2/datasets/HOI-M3/keypoint_reproj_viz/<seq>.mp4 (10 fps, 6x7 grid, libx264)
"""

import argparse
import json
import os
import os.path as osp
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np


# ------------------------------------------------------------------ paths
ROOT_KPS    = "/simurgh2/datasets/HOI-M3/mhr_simplified"
ROOT_IMG    = "/simurgh2/datasets/HOI-M3/images"
ROOT_CALIB  = "/simurgh2/datasets/HOI-M3/calib_ground_refined"
ROOT_OUT    = "/simurgh2/datasets/HOI-M3/keypoint_reproj_viz"
DATASET_INFO = "/simurgh/group/juze/datasets/HOI-M3/dataset_information.json"
SEQ_UNAVAIL  = "/simurgh2/datasets/HOI-M3/seq_unavail_views.json"

# Grid layout: 6 rows x 7 cols = 42 cells
GRID_ROWS, GRID_COLS = 6, 7
CELL_W, CELL_H = 640, 360   # half of 1280x720
N_VIEWS = 42

# MHR70 skeleton edges (COCO17-style for joints 0-16):
#   0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear,
#   5 left_shoulder, 6 right_shoulder, 7 left_elbow, 8 right_elbow,
#   9 left_wrist, 10 right_wrist, 11 left_hip, 12 right_hip,
#   13 left_knee, 14 right_knee, 15 left_ankle, 16 right_ankle
BODY_EDGES = [
    (5, 7), (7, 9), (6, 8), (8, 10),         # arms
    (5, 6), (5, 11), (6, 12), (11, 12),      # torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # legs
    (0, 1), (0, 2), (1, 3), (2, 4),          # head
]

# Per-person colors (BGR)
COLORS = [
    (60, 180, 75), (245, 130, 48), (0, 130, 200), (240, 50, 230),
    (210, 245, 60), (250, 190, 190), (0, 128, 128), (230, 190, 255),
    (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195),
]


# ------------------------------------------------------------------ helpers
def load_dataset_info():
    info = json.load(open(DATASET_INFO))
    seq2date = {}
    for date, seqs in info.items():
        for s in seqs:
            seq2date[s] = date
    return seq2date


def load_unavail():
    if not osp.isfile(SEQ_UNAVAIL):
        return {}
    d = json.load(open(SEQ_UNAVAIL))
    return d.get("seq_unavail_views", {})


def load_calib(date, target_w, target_h):
    """Load 42-view calib, scaling K from native imgSize -> target image size."""
    cf = osp.join(ROOT_CALIB, date, "calibration.json")
    raw = json.load(open(cf))
    cams = {}
    for k, v in raw.items():
        try:
            K  = np.asarray(v["K"], dtype=np.float64).reshape(3, 3)
            RT = np.asarray(v["RT"], dtype=np.float64).reshape(-1, 4)
            if RT.shape[0] == 4:
                RT = RT[:3]                              # 3x4
            dist = np.asarray(v.get("distCoeff", [0]*5), dtype=np.float64).reshape(-1)
            sw, sh = v.get("imgSize", [target_w, target_h])
            sx = float(target_w) / float(sw)
            sy = float(target_h) / float(sh)
            K = K.copy()
            K[0, 0] *= sx; K[0, 2] *= sx
            K[1, 1] *= sy; K[1, 2] *= sy
            cams[int(k)] = dict(K=K, R=RT[:3, :3], t=RT[:3, 3], dist=dist)
        except Exception:
            cams[int(k)] = None
    return cams


def project_world_to_image(pts3d, cam):
    """pts3d: (N,3) world. Returns (N,2) image pixels."""
    rvec, _ = cv2.Rodrigues(cam["R"])
    tvec = cam["t"].reshape(3, 1)
    pts2d, _ = cv2.projectPoints(
        pts3d.reshape(-1, 1, 3).astype(np.float64),
        rvec, tvec, cam["K"], cam["dist"].astype(np.float64),
    )
    return pts2d.reshape(-1, 2)


def draw_skeleton(img, kp2d, color, thickness=2, radius=3):
    h, w = img.shape[:2]
    pts = kp2d.astype(int)
    # joints
    for x, y in pts:
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(img, (x, y), radius, color, -1)
    # edges
    for a, b in BODY_EDGES:
        if a < len(pts) and b < len(pts):
            xa, ya = pts[a]; xb, yb = pts[b]
            if 0 <= xa < w and 0 <= ya < h and 0 <= xb < w and 0 <= yb < h:
                cv2.line(img, (xa, ya), (xb, yb), color, thickness, cv2.LINE_AA)


def render_cell(view_idx, fid, persons, cam, unavail_set, seq):
    """Return an (CELL_H, CELL_W, 3) BGR uint8 image for this view+frame."""
    cell = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
    if view_idx in unavail_set:
        cv2.rectangle(cell, (0, 0), (CELL_W, CELL_H), (0, 0, 60), -1)
        cv2.putText(cell, "UNAVAILABLE", (40, CELL_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(cell, f"v{view_idx:02d}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        return cell

    img_path = osp.join(ROOT_IMG, seq, str(view_idx), f"{fid:06d}.jpg")
    if not osp.isfile(img_path):
        cv2.putText(cell, f"v{view_idx:02d} no image", (10, CELL_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
        return cell

    img = cv2.imread(img_path)
    if img is None:
        cv2.putText(cell, f"v{view_idx:02d} read fail", (10, CELL_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
        return cell

    h, w = img.shape[:2]
    if cam is None:
        cv2.putText(img, "no calib", (10, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        for pid, p in enumerate(persons):
            kp = np.asarray(p, dtype=np.float64)         # (J, 3)
            if kp.size == 0:
                continue
            kp2d = project_world_to_image(kp[:, :3], cam)
            draw_skeleton(img, kp2d, COLORS[pid % len(COLORS)])

    cv2.putText(img, f"v{view_idx:02d}", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cell = cv2.resize(img, (CELL_W, CELL_H), interpolation=cv2.INTER_AREA)
    return cell


def render_grid_frame(fid, seq, cams, unavail_set, persons):
    grid = np.zeros((GRID_ROWS * CELL_H, GRID_COLS * CELL_W, 3), dtype=np.uint8)
    cells = [None] * N_VIEWS
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {
            ex.submit(render_cell, v, fid, persons, cams.get(v), unavail_set, seq): v
            for v in range(N_VIEWS)
        }
        for fu in as_completed(futs):
            v = futs[fu]
            cells[v] = fu.result()
    for v in range(N_VIEWS):
        r, c = divmod(v, GRID_COLS)
        grid[r*CELL_H:(r+1)*CELL_H, c*CELL_W:(c+1)*CELL_W] = cells[v]

    # frame stamp
    cv2.putText(grid, f"{seq}  fid={fid:06d}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)
    return grid


def load_persons(kps_path):
    try:
        d = json.load(open(kps_path))
    except Exception:
        return []
    out = []
    for p in d:
        kp = p.get("keypoints3d", [])
        if not kp:
            continue
        arr = np.asarray(kp, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] >= 3:
            out.append(arr[:, :3])
    return out


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--fps_in", type=int, default=30)
    ap.add_argument("--fps_out", type=int, default=10)
    ap.add_argument("--max_frames", type=int, default=0,
                    help="if >0, cap number of sampled frames (debug)")
    args = ap.parse_args()

    seq = args.seq
    out_mp4 = osp.join(ROOT_OUT, f"{seq}.mp4")
    os.makedirs(ROOT_OUT, exist_ok=True)

    if osp.isfile(out_mp4) and osp.getsize(out_mp4) > 10 * 1024 * 1024:
        print(f"[skip] {out_mp4} already exists ({osp.getsize(out_mp4)/1e6:.1f} MB)")
        return 0

    seq2date = load_dataset_info()
    if seq not in seq2date:
        print(f"[err] seq {seq} not in dataset_information.json", file=sys.stderr)
        return 2
    date = seq2date[seq]

    # discover sample image to get target size
    target_w, target_h = 1280, 720
    sample = None
    for v in range(N_VIEWS):
        cand = osp.join(ROOT_IMG, seq, str(v), "000000.jpg")
        if osp.isfile(cand):
            sample = cv2.imread(cand)
            if sample is not None:
                target_h, target_w = sample.shape[:2]
                break
    print(f"[info] target image size: {target_w}x{target_h}")

    cams = load_calib(date, target_w, target_h)
    unavail_set = set(load_unavail().get(seq, []))
    print(f"[info] seq={seq} date={date}  unavail views={sorted(unavail_set)}  ncams={sum(1 for v in cams.values() if v is not None)}")

    kps_dir = osp.join(ROOT_KPS, seq, "keypoints3d")
    if not osp.isdir(kps_dir):
        print(f"[err] no keypoints3d dir: {kps_dir}", file=sys.stderr)
        return 2

    fids = sorted(int(f.split(".")[0]) for f in os.listdir(kps_dir) if f.endswith(".json"))
    if not fids:
        print("[err] no keypoint frames", file=sys.stderr)
        return 2

    stride = max(1, args.fps_in // args.fps_out)
    sampled = fids[::stride]
    if args.max_frames > 0:
        sampled = sampled[: args.max_frames]
    print(f"[info] frames total={len(fids)} stride={stride} sampled={len(sampled)}")

    # ffmpeg pipe writer
    tmp_dir = tempfile.mkdtemp(prefix=f"reproj_{seq}_", dir="/tmp")
    print(f"[info] tmp dir = {tmp_dir}")
    grid_w = GRID_COLS * CELL_W
    grid_h = GRID_ROWS * CELL_H

    ff_cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{grid_w}x{grid_h}", "-pix_fmt", "bgr24",
        "-r", str(args.fps_out),
        "-i", "-",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        out_mp4,
    ]
    print("[ffmpeg]", " ".join(ff_cmd))
    proc = subprocess.Popen(ff_cmd, stdin=subprocess.PIPE)

    try:
        for i, fid in enumerate(sampled):
            persons = load_persons(osp.join(kps_dir, f"{fid:06d}.json"))
            grid = render_grid_frame(fid, seq, cams, unavail_set, persons)
            proc.stdin.write(grid.tobytes())
            if (i + 1) % 50 == 0 or i == 0:
                print(f"  frame {i+1}/{len(sampled)} fid={fid}")
        proc.stdin.close()
        rc = proc.wait()
        if rc != 0:
            print(f"[err] ffmpeg exit {rc}", file=sys.stderr)
            return rc
    finally:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

    if osp.isfile(out_mp4):
        print(f"[done] {out_mp4} {osp.getsize(out_mp4)/1e6:.1f} MB")
    else:
        print(f"[err] mp4 missing: {out_mp4}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
