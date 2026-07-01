#!/usr/bin/env python3
"""Step 3: Run YOLO11-seg inference on test sequences and save visualisations.

For each test sequence and selected views, runs the trained model on every Nth
frame, draws detections (bbox + class label + segmentation mask overlay), saves
annotated JPEG frames and compiles them into an MP4 video.

Usage:
    python scripts/yolo_seg/predict_visualize.py \
        --model /simurgh2/datasets/HOI-M3/yolo_seg_wds/runs/best.pt \
        --views 0 5 10 15 20 --step 60 --conf 0.25
"""

import argparse
import os
import sys

import cv2
from tqdm import tqdm

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

from ultralytics import YOLO  # noqa: E402
from scripts.visualize_mask_validity import preload_video_frames  # noqa: E402
from scripts.yolo_seg.config import (  # noqa: E402
    VIDEO_ROOT, WDS_ROOT, FRAME_STEP, IMGSZ,
    load_sequence_contents, get_test_sequences,
)


def parse_args():
    p = argparse.ArgumentParser(description="YOLO-seg inference + visualisation")
    p.add_argument("--model", type=str, required=True, help="Path to trained .pt weights")
    p.add_argument("--views", type=int, nargs="+", default=[0],
                   help="Camera views to visualise")
    p.add_argument("--step", type=int, default=FRAME_STEP, help="Frame step")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--imgsz", type=int, default=IMGSZ)
    p.add_argument("--output", type=str, default=os.path.join(WDS_ROOT, "predictions"))
    p.add_argument("--sequences", nargs="+", default=None,
                   help="Specific test sequences (default: all)")
    p.add_argument("--fps", type=int, default=10, help="Output video FPS")
    return p.parse_args()


def _frame_count(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def main():
    args = parse_args()
    model = YOLO(args.model)

    seq_contents = load_sequence_contents()
    test_seqs = get_test_sequences(seq_contents)
    if args.sequences:
        allowed = set(args.sequences)
        test_seqs = [s for s in test_seqs if s in allowed]

    print(f"Test sequences: {len(test_seqs)}, views: {args.views}")

    for seq_name in test_seqs:
        print(f"\n{'=' * 60}\n{seq_name}")
        videos_dir = os.path.join(VIDEO_ROOT, seq_name, "videos")
        if not os.path.isdir(videos_dir):
            print("  No videos directory, skipping")
            continue

        for view in args.views:
            video_path = os.path.join(videos_dir, f"{view}.mp4")
            if not os.path.exists(video_path):
                print(f"  View {view}: video not found")
                continue

            total = _frame_count(video_path)
            indices = list(range(0, total, args.step))
            tgt_w = args.imgsz
            tgt_h = int(tgt_w * 9 / 16)
            frames = preload_video_frames(video_path, indices, (tgt_w, tgt_h))
            if not frames:
                continue

            # Output paths
            frame_dir = os.path.join(args.output, seq_name, f"view_{view}")
            os.makedirs(frame_dir, exist_ok=True)

            annotated_frames = []
            for fid in tqdm(sorted(frames.keys()), desc=f"  view {view}", leave=False):
                results = model.predict(
                    frames[fid], imgsz=args.imgsz, conf=args.conf, verbose=False,
                )
                ann = results[0].plot()
                cv2.imwrite(os.path.join(frame_dir, f"frame_{fid:06d}.jpg"), ann)
                annotated_frames.append(ann)

            # Compile video
            if annotated_frames:
                h, w = annotated_frames[0].shape[:2]
                vid_path = os.path.join(args.output, seq_name, f"view_{view}.mp4")
                writer = cv2.VideoWriter(
                    vid_path, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h),
                )
                for af in annotated_frames:
                    writer.write(af)
                writer.release()
                print(f"  View {view}: {len(annotated_frames)} frames → {vid_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
