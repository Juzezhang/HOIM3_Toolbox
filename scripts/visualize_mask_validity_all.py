"""
Batch runner for visualization across all sequences.

This script wraps `scripts/visualize_mask_validity.py` and adds:
1) Auto discovery of sequences from validity output folders
2) Skip logic for already-generated videos
"""
import argparse
import os
import subprocess
import sys
from os.path import join
from typing import List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_VIEWS = [0, 2, 5, 6, 7, 8, 10, 11, 14, 15]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run visualize_mask_validity.py for all sequences with skip-if-done support."
    )
    parser.add_argument("--root_path", type=str, required=True,
                        help="Root path to HOI-M3 dataset")
    parser.add_argument("--validity_path", type=str, required=True,
                        help="Path to mask_validity root")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output root for visualization videos")

    # Sequence selection
    parser.add_argument("--seq_names", type=str, nargs="+", default=None,
                        help="Optional sequence whitelist")
    parser.add_argument("--seq_file", type=str, default=None,
                        help="Optional text file with one seq_name per line")
    parser.add_argument("--max_sequences", type=int, default=0,
                        help="Limit number of sequences (0 = no limit)")

    # Keep in sync with visualization script's common options
    parser.add_argument("--views", type=int, nargs="+", default=DEFAULT_VIEWS,
                        help="View indices to visualize")
    parser.add_argument("--all_views", action="store_true",
                        help="Use all 42 views (overrides --views)")
    parser.add_argument("--object_name", type=str, default=None,
                        help="Specific object to visualize (default: all)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=0)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--combined", action="store_true",
                        help="Create combined multi-view video")

    # Batch-run controls
    parser.add_argument("--skip_existing", dest="skip_existing", action="store_true",
                        help="Skip sequences whose expected videos already exist (default)")
    parser.add_argument("--no_skip_existing", dest="skip_existing", action="store_false",
                        help="Do not skip any sequence")
    parser.set_defaults(skip_existing=True)
    parser.add_argument("--dry_run", action="store_true",
                        help="Print actions without launching child processes")
    parser.add_argument("--stop_on_error", action="store_true",
                        help="Stop immediately if any sequence fails")
    return parser.parse_args()


def read_sequence_file(seq_file: str) -> List[str]:
    seqs: List[str] = []
    with open(seq_file, "r") as f:
        for line in f:
            seq = line.strip()
            if not seq or seq.startswith("#"):
                continue
            seqs.append(seq)
    return seqs


def discover_sequences(validity_path: str) -> List[str]:
    if not os.path.isdir(validity_path):
        return []

    seqs = []
    for name in sorted(os.listdir(validity_path)):
        seq_dir = join(validity_path, name)
        if not os.path.isdir(seq_dir):
            continue
        has_npz = any(f.endswith(".npz") for f in os.listdir(seq_dir))
        if has_npz:
            seqs.append(name)
    return seqs


def resolve_sequences(args) -> List[str]:
    if args.seq_names:
        seqs = list(dict.fromkeys(args.seq_names))
    elif args.seq_file:
        seqs = list(dict.fromkeys(read_sequence_file(args.seq_file)))
    else:
        seqs = discover_sequences(args.validity_path)

    if args.max_sequences > 0:
        seqs = seqs[:args.max_sequences]
    return seqs


def get_frame_files(validity_seq_path: str, start_frame: int, end_frame: int, step: int) -> List[str]:
    frame_files = sorted([f for f in os.listdir(validity_seq_path) if f.endswith(".npz")])
    if start_frame > 0 or end_frame > 0:
        frame_files = [
            f for f in frame_files
            if (start_frame <= int(f.replace(".npz", "")))
            and (int(f.replace(".npz", "")) < end_frame or end_frame == 0)
        ]
    if step > 1:
        frame_files = frame_files[::step]
    return frame_files


def get_object_names(validity_seq_path: str, frame_files: Sequence[str], object_name: Optional[str]) -> List[str]:
    if not frame_files:
        return []

    first_path = join(validity_seq_path, frame_files[0])
    data = np.load(first_path)
    try:
        objects = [k.replace("_validity", "") for k in data.keys() if k.endswith("_validity")]
    finally:
        data.close()

    objects = sorted(objects)
    if object_name:
        return [object_name] if object_name in objects else []
    return objects


def expected_video_paths(args, seq_name: str) -> Optional[List[str]]:
    validity_seq_path = join(args.validity_path, seq_name)
    if not os.path.isdir(validity_seq_path):
        return None

    frame_files = get_frame_files(validity_seq_path, args.start_frame, args.end_frame, args.step)
    if not frame_files:
        return []

    objects = get_object_names(validity_seq_path, frame_files, args.object_name)
    if not objects:
        return []

    out_seq = join(args.output_path, seq_name)
    views = list(range(42)) if args.all_views else args.views

    if args.combined:
        return [join(out_seq, f"{obj}_combined.mp4") for obj in objects]

    expected = []
    for obj in objects:
        for view_idx in views:
            expected.append(join(out_seq, f"{obj}_view{view_idx}.mp4"))
    return expected


def videos_done(expected_paths: Sequence[str]) -> bool:
    if len(expected_paths) == 0:
        return True
    for p in expected_paths:
        if not os.path.exists(p):
            return False
        if os.path.getsize(p) == 0:
            return False
    return True


def build_child_command(args, seq_name: str) -> List[str]:
    cmd = [
        sys.executable,
        "scripts/visualize_mask_validity.py",
        "--root_path", args.root_path,
        "--seq_name", seq_name,
        "--validity_path", args.validity_path,
        "--output_path", args.output_path,
        "--fps", str(args.fps),
        "--start_frame", str(args.start_frame),
        "--end_frame", str(args.end_frame),
        "--step", str(args.step),
        "--alpha", str(args.alpha),
    ]

    if args.all_views:
        cmd.append("--all_views")
    else:
        cmd.extend(["--views"] + [str(v) for v in args.views])

    if args.object_name:
        cmd.extend(["--object_name", args.object_name])
    if args.combined:
        cmd.append("--combined")
    return cmd


def run_one_sequence(args, seq_name: str) -> Tuple[bool, str]:
    expected = expected_video_paths(args, seq_name)
    if expected is None:
        return True, f"[{seq_name}] skip: missing validity folder"
    if len(expected) == 0:
        return True, f"[{seq_name}] skip: nothing to visualize"

    if args.skip_existing and videos_done(expected):
        return True, f"[{seq_name}] skip: already complete ({len(expected)} videos)"

    cmd = build_child_command(args, seq_name)
    if args.dry_run:
        return True, f"[{seq_name}] dry-run: {' '.join(cmd)}"

    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        return False, f"[{seq_name}] FAILED (exit={ret.returncode})"
    return True, f"[{seq_name}] done"


def main():
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    seqs = resolve_sequences(args)
    print(f"Found {len(seqs)} sequences to consider")

    ok_count = 0
    fail_count = 0
    skip_count = 0
    failed: List[str] = []

    for i, seq_name in enumerate(seqs, start=1):
        print(f"\n[{i}/{len(seqs)}] {seq_name}")
        ok, msg = run_one_sequence(args, seq_name)
        print(msg)

        if "skip:" in msg:
            skip_count += 1
            continue

        if ok:
            ok_count += 1
        else:
            fail_count += 1
            failed.append(seq_name)
            if args.stop_on_error:
                break

    print("\n=== Summary ===")
    print(f"processed_ok: {ok_count}")
    print(f"skipped: {skip_count}")
    print(f"failed: {fail_count}")
    if failed:
        print("failed_sequences:")
        for seq in failed:
            print(f"  - {seq}")
        sys.exit(1)


if __name__ == "__main__":
    main()
