"""Benchmark Cutie vs SAM3 video propagation speed on the same task:
single-view 1080p video, propagate object mask from frame 0 across N frames.

Uses office_data32 view 0 as test sequence.
"""
import argparse
import os
import sys
import time
import tempfile
import shutil

import cv2
import numpy as np


def bench_sam3_video(video_dir, prompt, n_frames):
    """Run SAM3 video predictor with text prompt, time propagation."""
    sys.path.insert(0, "/simurgh/u/juze/code/sam3")
    print("[SAM3] loading model...", flush=True)
    t_load = time.time()
    from sam3.model_builder import build_sam3_video_predictor
    predictor = build_sam3_video_predictor()
    print(f"[SAM3] model loaded in {time.time()-t_load:.1f}s", flush=True)

    t_session = time.time()
    resp = predictor.handle_request(dict(type="start_session", resource_path=video_dir))
    sid = resp["session_id"]
    print(f"[SAM3] session start in {time.time()-t_session:.1f}s", flush=True)

    t_prompt = time.time()
    predictor.handle_request(dict(type="add_prompt", session_id=sid,
                                  frame_index=0, text=prompt))
    print(f"[SAM3] add text prompt in {time.time()-t_prompt:.1f}s", flush=True)

    t_prop = time.time()
    n_done = 0
    for stream in predictor.handle_stream_request(
        dict(type="propagate_in_video", session_id=sid)
    ):
        n_done += 1
        if n_done >= n_frames:
            break
    dt = time.time() - t_prop
    return n_done, dt


def bench_cutie(video_dir, n_frames):
    """Run Cutie tracking — load existing reference mask (from cutie_refs)."""
    sys.path.insert(0, "/simurgh/u/juze/code/HHOI-Toolkit/Trackingmask")
    print("[Cutie] loading model...", flush=True)
    t_load = time.time()
    import torch
    from cutie.inference.inference_core import InferenceCore
    from cutie.model.cutie import CUTIE
    from omegaconf import OmegaConf
    # Try matching existing cutie_track_one_view.py config
    cfg = OmegaConf.load("/simurgh/u/juze/code/HHOI-Toolkit/Trackingmask/cutie/config/eval_config.yaml")
    network = CUTIE(cfg).cuda().eval()
    weights = torch.load(cfg.weights, map_location="cuda")
    network.load_weights(weights)
    print(f"[Cutie] model loaded in {time.time()-t_load:.1f}s", flush=True)

    # Need a reference mask — use office_data32 v0 cutie_ref
    ref_npy = "/simurgh2/datasets/HOI-M3/cutie_refs/office_data32/masks/0.npy"
    ref = np.load(ref_npy)  # indexed mask
    # Convert to one-hot per object
    n_objs = ref.max()
    ref_oh = np.zeros((n_objs, *ref.shape), dtype=np.float32)
    for i in range(n_objs):
        ref_oh[i] = (ref == i + 1).astype(np.float32)
    ref_torch = torch.from_numpy(ref_oh).cuda()

    processor = InferenceCore(network, cfg=cfg)
    processor.max_internal_size = 480

    files = sorted([f for f in os.listdir(video_dir) if f.endswith(".jpg")])[:n_frames]
    t_prop = time.time()
    n_done = 0
    for i, fname in enumerate(files):
        img = cv2.imread(os.path.join(video_dir, fname))
        img_torch = torch.from_numpy(img[:, :, ::-1].copy()).cuda().permute(2, 0, 1).float() / 255.0
        if i == 0:
            processor.step(img_torch, ref_torch, idx_mask=False)
        else:
            processor.step(img_torch)
        n_done += 1
    dt = time.time() - t_prop
    return n_done, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sam3", "cutie", "both"], default="sam3")
    ap.add_argument("--seq", default="office_data32")
    ap.add_argument("--view", type=int, default=0)
    ap.add_argument("--n_frames", type=int, default=100)
    ap.add_argument("--start_frame", type=int, default=0)
    ap.add_argument("--prompt", default="person")
    args = ap.parse_args()

    # Materialize n_frames as video dir
    tmp = tempfile.mkdtemp(prefix="bench_")
    img_root = f"/simurgh2/datasets/HOI-M3/images/{args.seq}/{args.view}"
    print(f"[bench] preparing {args.n_frames} frames in {tmp}")
    for i, fid in enumerate(range(args.start_frame, args.start_frame + args.n_frames)):
        src = os.path.join(img_root, f"{fid:06d}.jpg")
        if not os.path.isfile(src):
            print(f"[bench] missing {src}, abort")
            return 1
        dst = os.path.join(tmp, f"{i:05d}.jpg")
        shutil.copy(src, dst)
    print("[bench] frames ready")

    if args.mode in ("sam3", "both"):
        print(f"\n=== SAM3 bench (prompt='{args.prompt}') ===")
        n, dt = bench_sam3_video(tmp, args.prompt, args.n_frames)
        print(f"\n*** SAM3: {n} frames in {dt:.1f}s = {n/dt:.2f} fps ***\n")

    if args.mode in ("cutie", "both"):
        print(f"\n=== Cutie bench ===")
        n, dt = bench_cutie(tmp, args.n_frames)
        print(f"\n*** Cutie: {n} frames in {dt:.1f}s = {n/dt:.2f} fps ***\n")

    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
