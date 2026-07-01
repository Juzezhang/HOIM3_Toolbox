"""
Batch runner for multi-view mask validity checking across all sequences.

This script wraps `scripts/multi_view_mask_check.py` and adds:
1) Auto discovery of all sequences
2) Skip logic for already-processed sequences
"""
import argparse
import json
import os
import subprocess
import sys
import time
from os.path import join
from typing import List, Optional, Sequence, Tuple


# Allow importing repository modules when running as:
# python scripts/multi_view_mask_check_all.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.utils.mask_io import detect_mask_format  # noqa: E402


DEFAULT_VIEWS = [0, 2, 5, 6, 7, 8, 10, 11, 14, 15]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run multi_view_mask_check.py for all sequences with skip-if-done support."
    )
    parser.add_argument("--root_path", type=str, required=True,
                        help="Root path to HOI-M3 dataset")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output root for validity results")

    # Sequence selection
    parser.add_argument("--seq_names", type=str, nargs="+", default=None,
                        help="Optional sequence whitelist")
    parser.add_argument("--seq_file", type=str, default=None,
                        help="Optional text file with one seq_name per line")
    parser.add_argument("--max_sequences", type=int, default=0,
                        help="Limit number of sequences (0 = no limit)")

    # Keep in sync with core script's common options
    parser.add_argument("--views", type=int, nargs="+", default=DEFAULT_VIEWS,
                        help="View indices to process")
    parser.add_argument("--all_views", action="store_true",
                        help="Use all available views from calibration")
    parser.add_argument("--voxel_res", type=int, default=48)
    parser.add_argument("--max_iters", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--thresh_init", type=float, default=0.5)
    parser.add_argument("--thresh_expand", type=float, default=0.35)
    parser.add_argument("--prec_penalty", type=float, default=0.3)
    parser.add_argument("--area_penalty", type=float, default=0.05)
    parser.add_argument("--bbox_padding", type=float, default=0.30)
    parser.add_argument("--min_overlap", type=float, default=0.02)
    parser.add_argument("--mask_format", type=str, default="auto",
                        choices=["auto", "npz", "shard"])
    parser.add_argument("--mask_root", type=str, default=None,
                        help="Root directory for shard masks")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--verbose", action="store_true")

    # Batch-run controls
    parser.add_argument("--skip_existing", dest="skip_existing", action="store_true",
                        help="Skip sequences whose expected outputs already exist (default)")
    parser.add_argument("--no_skip_existing", dest="skip_existing", action="store_false",
                        help="Do not skip any sequence")
    parser.set_defaults(skip_existing=True)
    parser.add_argument("--dry_run", action="store_true",
                        help="Print actions without launching child processes")
    parser.add_argument("--stop_on_error", action="store_true",
                        help="Stop immediately if any sequence fails")
    return parser.parse_args()


def load_all_sequences(root_path: str) -> List[str]:
    info_path = join(root_path, "dataset_information.json")
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"Missing dataset metadata: {info_path}")

    with open(info_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("dataset_information.json must be a date->sequence-list dict")

    seqs: List[str] = []
    seen = set()
    for date_key in sorted(data.keys()):
        seq_list = data[date_key]
        if not isinstance(seq_list, list):
            continue
        for seq in seq_list:
            if isinstance(seq, str) and seq not in seen:
                seqs.append(seq)
                seen.add(seq)
    return seqs


def read_sequence_file(seq_file: str) -> List[str]:
    seqs: List[str] = []
    with open(seq_file, "r") as f:
        for line in f:
            seq = line.strip()
            if not seq or seq.startswith("#"):
                continue
            seqs.append(seq)
    return seqs


def resolve_sequences(args) -> List[str]:
    if args.seq_names:
        seqs = list(dict.fromkeys(args.seq_names))
    elif args.seq_file:
        seqs = list(dict.fromkeys(read_sequence_file(args.seq_file)))
    else:
        seqs = load_all_sequences(args.root_path)

    if args.max_sequences > 0:
        seqs = seqs[:args.max_sequences]
    return seqs


def _resolve_mask_format_for_seq(args, seq_name: str) -> str:
    if args.mask_format != "auto":
        return args.mask_format
    return detect_mask_format(args.root_path, seq_name, args.mask_root)


def _expected_outputs_shard(args, seq_name: str) -> Optional[List[str]]:
    shard_root = args.mask_root if args.mask_root else join(args.root_path, "mask_shards")
    meta_path = join(shard_root, seq_name, "meta.json")
    if not os.path.exists(meta_path):
        return None

    with open(meta_path, "r") as f:
        meta = json.load(f)

    frame_ids = meta.get("frame_ids", [])
    if not frame_ids:
        return []

    frame_ids_int = [int(fid) for fid in frame_ids]
    if args.start_frame > 0 or args.end_frame > 0:
        start = args.start_frame
        end = args.end_frame if args.end_frame > 0 else (max(frame_ids_int) + 1)
        frame_ids_int = [fid for fid in frame_ids_int if start <= fid < end]

    return [f"{fid:06d}.npz" for fid in frame_ids_int]


def _expected_outputs_npz(args, seq_name: str) -> Optional[List[str]]:
    npz_dir = join(args.root_path, "mask_npz", seq_name)
    if not os.path.isdir(npz_dir):
        return None
    frame_files = sorted([f for f in os.listdir(npz_dir) if f.endswith(".npz")])
    if args.start_frame > 0 or args.end_frame > 0:
        start = args.start_frame
        end = args.end_frame if args.end_frame > 0 else len(frame_files)
        frame_files = frame_files[start:end]
    return frame_files


def expected_output_files(args, seq_name: str) -> Optional[List[str]]:
    mask_format = _resolve_mask_format_for_seq(args, seq_name)
    if mask_format == "shard":
        return _expected_outputs_shard(args, seq_name)
    return _expected_outputs_npz(args, seq_name)


def sequence_done(args, seq_name: str, expected_files: Sequence[str]) -> bool:
    out_seq = join(args.output_path, seq_name)
    if not os.path.isdir(out_seq):
        return False
    if len(expected_files) == 0:
        return True

    existing = set(f for f in os.listdir(out_seq) if f.endswith(".npz"))
    return all(fname in existing for fname in expected_files)


def build_child_command(args, seq_name: str) -> List[str]:
    cmd = [
        sys.executable,
        "scripts/multi_view_mask_check.py",
        "--root_path", args.root_path,
        "--seq_name", seq_name,
        "--output_path", args.output_path,
        "--voxel_res", str(args.voxel_res),
        "--max_iters", str(args.max_iters),
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--thresh_init", str(args.thresh_init),
        "--thresh_expand", str(args.thresh_expand),
        "--prec_penalty", str(args.prec_penalty),
        "--area_penalty", str(args.area_penalty),
        "--bbox_padding", str(args.bbox_padding),
        "--min_overlap", str(args.min_overlap),
        "--mask_format", args.mask_format,
        "--start_frame", str(args.start_frame),
        "--end_frame", str(args.end_frame),
        "--device", args.device,
    ]

    if args.mask_root:
        cmd.extend(["--mask_root", args.mask_root])

    if args.all_views:
        cmd.append("--all_views")
    else:
        cmd.extend(["--views"] + [str(v) for v in args.views])

    if args.verbose:
        cmd.append("--verbose")
    return cmd


def run_one_sequence(args, seq_name: str) -> Tuple[bool, str]:
    expected = expected_output_files(args, seq_name)
    if expected is None:
        return True, f"[{seq_name}] skip: missing input masks"
    if len(expected) == 0:
        return True, f"[{seq_name}] skip: nothing to process"

    if args.skip_existing and sequence_done(args, seq_name, expected):
        return True, f"[{seq_name}] skip: already complete ({len(expected)} frames)"

    cmd = build_child_command(args, seq_name)
    if args.dry_run:
        return True, f"[{seq_name}] dry-run: {' '.join(cmd)}"

    t0 = time.time()
    ret = subprocess.run(cmd)
    dt = time.time() - t0
    if ret.returncode != 0:
        return False, f"[{seq_name}] FAILED (exit={ret.returncode}, {dt:.1f}s)"
    return True, f"[{seq_name}] done ({dt:.1f}s)"


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
