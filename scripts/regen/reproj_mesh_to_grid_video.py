#!/usr/bin/env python3
"""Render fitted MHR mesh on 42-view grid mp4 (per HOI-M3 sequence).

Input:
  - /simurgh2/datasets/HOI-M3/mhr_simplified/<seq>/mhr/<fid:06d>.json
        list of persons with {model_parameters, identity_coeffs, face_expr_coeffs, Rh, Th}
  - /simurgh2/datasets/HOI-M3/calib_ground_refined/<date>/calibration.json
  - /simurgh2/datasets/HOI-M3/images/<seq>/<view>/<fid:06d>.jpg

Output:
  - /simurgh2/datasets/HOI-M3/mesh_reproj_viz/<seq>.mp4 (10 fps, 6x7 grid)
"""
import argparse
import json
import os
import os.path as osp
import subprocess
import sys
import tempfile

import cv2
import numpy as np

# ------------------------------------------------------------------ paths
ROOT_FIT    = "/simurgh2/datasets/HOI-M3/mhr_simplified"
ROOT_IMG    = "/simurgh2/datasets/HOI-M3/images"
ROOT_CALIB  = "/simurgh2/datasets/HOI-M3/calib_ground_refined"
ROOT_OUT    = "/simurgh2/datasets/HOI-M3/mesh_reproj_viz"
DATASET_INFO = "/simurgh/group/juze/datasets/HOI-M3/dataset_information.json"
SEQ_UNAVAIL  = "/simurgh2/datasets/HOI-M3/seq_unavail_views.json"

GRID_ROWS, GRID_COLS = 6, 7
CELL_W, CELL_H = 640, 360
N_VIEWS = 42


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
    cf = osp.join(ROOT_CALIB, date, "calibration.json")
    raw = json.load(open(cf))
    cams = {}
    for k, v in raw.items():
        try:
            K  = np.asarray(v["K"], dtype=np.float64).reshape(3, 3)
            RT = np.asarray(v["RT"], dtype=np.float64).reshape(-1, 4)
            if RT.shape[0] == 4: RT = RT[:3]
            sw, sh = v.get("imgSize", [target_w, target_h])
            sx = float(target_w) / float(sw)
            sy = float(target_h) / float(sh)
            K = K.copy()
            K[0, 0] *= sx; K[0, 2] *= sx
            K[1, 1] *= sy; K[1, 2] *= sy
            cams[int(k)] = dict(K=K, R=RT[:3, :3].astype(np.float64), T=RT[:3, 3:4].astype(np.float64))
        except Exception:
            cams[int(k)] = None
    return cams


def load_persons_fit(json_path):
    """Returns list of dicts: {model_parameters, identity_coeffs, face_expr_coeffs}.
    Keeps batch dim (1, D) — model_forward expects this."""
    try:
        d = json.load(open(json_path))
    except Exception:
        return []
    out = []
    for p in d:
        try:
            out.append({
                "id": p.get("id", 0),
                "model_parameters": np.array(p["model_parameters"], dtype=np.float32),
                "identity_coeffs": np.array(p["identity_coeffs"], dtype=np.float32),
                "face_expr_coeffs": np.array(p["face_expr_coeffs"], dtype=np.float32)
                    if "face_expr_coeffs" in p else np.zeros((1, 72), dtype=np.float32),
            })
        except Exception:
            continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--fps_in", type=int, default=30)
    ap.add_argument("--fps_out", type=int, default=10)
    ap.add_argument("--max_frames", type=int, default=300,
                    help="Cap rendered frames. 300 @ 10fps = 30s mp4 (default).")
    args = ap.parse_args()

    seq = args.seq
    out_mp4 = osp.join(ROOT_OUT, f"{seq}.mp4")
    os.makedirs(ROOT_OUT, exist_ok=True)

    if osp.isfile(out_mp4) and osp.getsize(out_mp4) > 10 * 1024 * 1024:
        print(f"[skip] {out_mp4} already exists")
        return 0

    # MHR model + renderer
    sys.path.insert(0, "/simurgh/u/juze/code/mv-bodyfit")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.chdir("/simurgh/u/juze/code/mv-bodyfit")
    print("[Vis] Loading MHR model...")
    from mvbodyfit.io.model import MHRLoader
    model_dict = MHRLoader(path="config/model/mhr.yml")()
    model_forward = model_dict["model"]
    faces = model_dict.get("faces", None)
    if faces is None:
        print("ERROR: no faces"); return 2

    from mvbodyfit.core.visualize.pyrender_wrapper import Renderer
    renderer = Renderer()

    seq2date = load_dataset_info()
    if seq not in seq2date:
        print(f"[err] seq {seq} not in dataset_information"); return 2
    date = seq2date[seq]

    target_w, target_h = 1280, 720
    for v in range(N_VIEWS):
        cand = osp.join(ROOT_IMG, seq, str(v), "000000.jpg")
        if osp.isfile(cand):
            sample = cv2.imread(cand)
            if sample is not None:
                target_h, target_w = sample.shape[:2]
                break
    print(f"[info] target size {target_w}x{target_h}")

    cams = load_calib(date, target_w, target_h)
    unavail_set = set(load_unavail().get(seq, []))

    mhr_dir = osp.join(ROOT_FIT, seq, "mhr")
    if not osp.isdir(mhr_dir):
        print(f"[err] no mhr dir: {mhr_dir}"); return 2

    fids = sorted(int(f.split(".")[0]) for f in os.listdir(mhr_dir) if f.endswith(".json"))
    stride = max(1, args.fps_in // args.fps_out)
    sampled = fids[::stride]
    if args.max_frames > 0:
        sampled = sampled[: args.max_frames]
    print(f"[info] total={len(fids)} stride={stride} sampled={len(sampled)}")

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
    proc = subprocess.Popen(ff_cmd, stdin=subprocess.PIPE)

    import torch
    import time
    t0 = time.time()

    for i, fid in enumerate(sampled):
        # Load persons + build mesh
        persons = load_persons_fit(osp.join(mhr_dir, f"{fid:06d}.json"))
        persons_verts = []
        for p in persons:
            try:
                params = {
                    "model_parameters": p["model_parameters"],
                    "identity_coeffs": p["identity_coeffs"],
                    "face_expr_coeffs": p["face_expr_coeffs"],
                }
                with torch.no_grad():
                    out = model_forward(params, ret_vertices=True, return_tensor=False)
                verts = out["vertices"][0]
                persons_verts.append({"vertices": verts, "pid": int(p["id"])})
            except Exception as e:
                if i == 0:
                    print(f"  build mesh fail pid={p['id']}: {e}", flush=True)

        # Render each view + composite grid
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        for v in range(N_VIEWS):
            cell = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
            if v in unavail_set:
                cv2.rectangle(cell, (0, 0), (CELL_W, CELL_H), (0, 0, 60), -1)
                cv2.putText(cell, "UNAVAIL", (40, CELL_H // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            else:
                img_path = osp.join(ROOT_IMG, seq, str(v), f"{fid:06d}.jpg")
                img = cv2.imread(img_path) if osp.isfile(img_path) else None
                if img is None:
                    img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                cam = cams.get(v)
                if cam is not None and persons_verts:
                    camera = {"K": cam["K"], "R": cam["R"], "T": cam["T"]}
                    # Render each person separately with bright color overlay.
                    PERSON_COLORS_BGR = [
                        (60, 60, 220),     # red
                        (80, 220, 80),     # green
                        (220, 200, 60),    # cyan-ish
                        (220, 60, 220),    # magenta
                        (60, 220, 220),    # yellow
                    ]
                    overlay = img.astype(np.float32)
                    for pv in persons_verts:
                        pid = pv["pid"]
                        render_data = {
                            pid: {
                                "vertices": pv["vertices"],
                                "faces": faces,
                                "vid": pid,
                                "name": f"person_{pid}",
                            }
                        }
                        try:
                            out = renderer.render_image(render_data, img, camera, [])
                            rend_rgba = np.asarray(out[0][0])  # (H, W, 4) RGBA on white bg
                            alpha = rend_rgba[..., 3].astype(np.float32) / 255.0  # 0..1
                            sel = alpha > 0.05
                            if sel.sum() > 0:
                                bright = np.asarray(
                                    PERSON_COLORS_BGR[pid % len(PERSON_COLORS_BGR)],
                                    dtype=np.float32,
                                )
                                MESH_ALPHA = 0.55
                                a = (alpha * MESH_ALPHA)[..., None]
                                overlay = overlay * (1.0 - a) + bright[None, None, :] * a
                        except Exception as e:
                            if i == 0:
                                print(f"  render fail v{v} pid={pid}: {e}", flush=True)
                    img = overlay.clip(0, 255).astype(np.uint8)
                cv2.putText(img, f"v{v:02d}", (8, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cell = cv2.resize(img, (CELL_W, CELL_H), interpolation=cv2.INTER_AREA)
            r, c = divmod(v, GRID_COLS)
            grid[r*CELL_H:(r+1)*CELL_H, c*CELL_W:(c+1)*CELL_W] = cell

        cv2.putText(grid, f"{seq}  fid={fid:06d}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)

        try:
            proc.stdin.write(grid.tobytes())
        except BrokenPipeError:
            print("[err] ffmpeg pipe broken"); break

        if (i + 1) % 5 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(sampled) - i - 1) / max(rate, 0.01)
            print(f"  {i+1}/{len(sampled)} ({rate:.2f} fr/s, ETA {eta/60:.1f} min)", flush=True)

    proc.stdin.close()
    proc.wait()
    print(f"[done] -> {out_mp4}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
