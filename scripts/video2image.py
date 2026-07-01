#!/usr/bin/env python3
"""Extract all HOI-M3 video frames to JPEG images at 1080p resolution.

For each sequence and each view (42 cameras), extracts every frame using ffmpeg
and saves as JPEG images.

Output structure:
    /simurgh2/datasets/HOI-M3/images/{seq_name}/{view_id}/{frame:06d}.jpg

Usage:
    python scripts/video2image.py
    python scripts/video2image.py --step 60                    # every 60th frame (1fps)
    python scripts/video2image.py --sequences bedroom_data01   # specific sequences
    python scripts/video2image.py --views 0 7 14               # specific views
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

VIDEO_ROOT = "/simurgh/group/juze/datasets/HOI-M3/videos"
OUTPUT_ROOT = "/simurgh2/datasets/HOI-M3/images"
TARGET_W, TARGET_H = 1280, 720  # 720p


def parse_args():
    p = argparse.ArgumentParser(description="Extract HOI-M3 video frames to images")
    p.add_argument("--step", type=int, default=1,
                   help="Frame step (1=all frames, 60=1fps, 600=0.1fps)")
    p.add_argument("--sequences", nargs="+", default=None,
                   help="Specific sequences to process (default: all)")
    p.add_argument("--views", type=int, nargs="+", default=None,
                   help="Specific views (default: all 42)")
    p.add_argument("--quality", type=int, default=95,
                   help="JPEG quality (1-100)")
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel ffmpeg processes")
    return p.parse_args()


def extract_view(video_path, output_dir, step, quality):
    """Extract frames from one video using ffmpeg."""
    os.makedirs(output_dir, exist_ok=True)

    # Quick check if already done (just test if first frame exists, avoid slow NFS listdir)
    if os.path.isfile(os.path.join(output_dir, "000000.jpg")):
        return -1  # skip, already extracted

    if step == 1:
        # Extract all frames
        vf = f"scale={TARGET_W}:{TARGET_H}"
    else:
        # Extract every Nth frame
        vf = f"select='not(mod(n\\,{step}))',scale={TARGET_W}:{TARGET_H}"

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", vf,
        "-vsync", "vfr",
        "-q:v", str(max(1, min(31, (100 - quality) * 31 // 100 + 1))),
        "-start_number", "0",
        os.path.join(output_dir, "%06d.jpg"),
        "-y",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    ERROR: {video_path}: {result.stderr[-200:]}", flush=True)
        return 0

    n = len([f for f in os.listdir(output_dir) if f.endswith(".jpg")])
    return n


def main():
    args = parse_args()

    # Get sequence list
    all_seqs = sorted(d for d in os.listdir(VIDEO_ROOT)
                      if os.path.isdir(os.path.join(VIDEO_ROOT, d)))
    if args.sequences:
        all_seqs = [s for s in all_seqs if s in set(args.sequences)]

    print(f"Sequences: {len(all_seqs)}")
    print(f"Resolution: {TARGET_W}x{TARGET_H}")
    print(f"Step: {args.step} ({'all frames' if args.step == 1 else f'every {args.step}th frame'})")
    print(f"Output: {OUTPUT_ROOT}")
    print()

    total_frames = 0
    for seq_idx, seq_name in enumerate(all_seqs):
        videos_dir = os.path.join(VIDEO_ROOT, seq_name, "videos")
        if not os.path.isdir(videos_dir):
            continue

        # Get view list
        view_files = sorted(
            [f for f in os.listdir(videos_dir) if f.endswith(".mp4")],
            key=lambda f: int(f.replace(".mp4", ""))
        )
        if args.views is not None:
            view_files = [f for f in view_files if int(f.replace(".mp4", "")) in args.views]

        print(f"[{seq_idx+1}/{len(all_seqs)}] {seq_name}: {len(view_files)} views", flush=True)

        def _process_view(vf):
            view_id = vf.replace(".mp4", "")
            video_path = os.path.join(videos_dir, vf)
            out_dir = os.path.join(OUTPUT_ROOT, seq_name, view_id)
            n = extract_view(video_path, out_dir, args.step, args.quality)
            return view_id, n

        seq_frames = 0
        skipped = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(_process_view, view_files))
        for view_id, n in results:
            if n == -1:
                skipped += 1
            else:
                seq_frames += n

        total_frames += seq_frames
        if skipped == len(view_files):
            print(f"  → skipped (already extracted)", flush=True)
        else:
            print(f"  → {seq_frames} frames, {skipped} views skipped", flush=True)

    print(f"\nDone. {total_frames:,} total frames in {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
