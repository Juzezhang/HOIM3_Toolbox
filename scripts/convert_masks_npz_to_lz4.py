"""
Convert mask data from NPZ (zlib) to object-major shard (LZ4+bitpack) format.

Each sequence produces:
  {dst_root}/mask_shards/{seq_name}/meta.json
  {dst_root}/mask_shards/{seq_name}/{object}.shard

Usage:
    python scripts/convert_masks_npz_to_lz4.py \
        --src_root /simurgh/group/juze/datasets/HOI-M3 \
        --dst_root /simurgh2/datasets/HOI-M3 \
        --sequences bedroom_data01 \
        --num_workers 12 --compression_level 9 --validate
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import zipfile
from os.path import join, isfile, isdir
from multiprocessing import Pool
from functools import partial

sys.path.insert(0, join(os.path.dirname(__file__), ".."))
from scripts.utils.mask_io import ShardWriter, ShardReader, compress_mask_frame


def _load_mask_auto(data, key):
    """Load a mask from an NPZ, auto-detecting bitpacked-vs-raw format.

    Bitpacked format (new aggregate_cutie_to_npz.py output):
        data[key]            -> (V, H, W_packed) uint8 (big-endian packbits along width)
        data[f"{key}__shape"] -> (3,) int64 = original (V, H, W)
    Returns reconstructed (V, H, W) uint8 with values {0, 255}.

    Legacy format (raw uncompressed):
        data[key]            -> (V, H, W) uint8 with values {0, 255}.
    """
    shape_key = f"{key}__shape"
    arr = data[key]
    if shape_key in data.files:
        original_shape = tuple(int(x) for x in data[shape_key])
        v, h, w = original_shape
        unpacked_bool = np.unpackbits(arr, axis=-1, count=w).astype(bool).reshape(v, h, w)
        return unpacked_bool.astype(np.uint8) * 255
    return arr


def _list_object_keys(npz_files):
    """Return object keys from an NPZ file list (filter out __shape sidecars)."""
    return sorted([k for k in npz_files if not k.endswith("__shape")])


def _compress_one_frame(args, compression_level=9, expected_objects=None, expected_shape=None):
    """Worker: read one NPZ, compress each object → {obj: compressed_bytes}.

    Runs in a subprocess (multiprocessing.Pool).
    Auto-detects bitpacked vs raw mask layout (see _load_mask_auto).
    """
    frame_file, mask_npz_path = args
    frame_id = int(os.path.splitext(frame_file)[0])
    npz_path = join(mask_npz_path, frame_file)
    try:
        data = np.load(npz_path)
    except (zipfile.BadZipFile, OSError, ValueError, EOFError) as e:
        return frame_id, frame_file, None, None, f"{type(e).__name__}: {e}"

    try:
        result = {}
        available_keys = set(data.files)
        objects = expected_objects if expected_objects is not None else _list_object_keys(data.files)
        for key in objects:
            if key not in available_keys:
                raise KeyError(f"Missing object '{key}' in {frame_file}")
            mask = _load_mask_auto(data, key)  # (views, H, W) uint8 {0,255}
            if expected_shape is not None and tuple(mask.shape) != tuple(expected_shape):
                raise ValueError(
                    f"Shape mismatch for '{key}' in {frame_file}: "
                    f"got {tuple(mask.shape)}, expected {tuple(expected_shape)}"
                )
            compressed = compress_mask_frame(mask, compression_level)
            result[key] = compressed

        # Shape info from first object (after unpacking)
        first_key = objects[0]
        shape = _load_mask_auto(data, first_key).shape  # (views, H, W)
        return frame_id, frame_file, result, shape, None
    except Exception as e:
        return frame_id, frame_file, None, None, f"{type(e).__name__}: {e}"
    finally:
        data.close()


def _find_first_valid_npz(mask_npz_path, frame_files):
    """Find first valid NPZ to infer object names and shape.

    Auto-handles bitpacked vs raw npz layouts.
    """
    for frame_file in frame_files:
        path = join(mask_npz_path, frame_file)
        try:
            data = np.load(path)
            objects = _list_object_keys(data.files)
            if not objects:
                data.close()
                continue
            # Use _load_mask_auto so shape reflects original (V,H,W) regardless of format.
            shape = _load_mask_auto(data, objects[0]).shape
            data.close()
            return frame_file, objects, shape
        except (zipfile.BadZipFile, OSError, ValueError, EOFError):
            continue
    return None, None, None


def convert_sequence(seq_name, src_root, dst_root, num_workers, compression_level):
    """Convert one sequence from NPZ to shard format."""
    mask_npz_path = join(src_root, "mask_npz", seq_name)
    dst_path = join(dst_root, "mask_shards", seq_name)

    # Check for resume
    meta_path = join(dst_path, "meta.json")
    if isfile(meta_path):
        print(f"  [SKIP] {seq_name}: meta.json already exists")
        return True

    # List NPZ frames
    frame_files = sorted([f for f in os.listdir(mask_npz_path) if f.endswith(".npz")])
    num_frames = len(frame_files)
    if num_frames == 0:
        print(f"  [SKIP] {seq_name}: no NPZ files found")
        return False

    print(f"  Converting {seq_name}: {num_frames} frames ...")

    # Peek at first *valid* frame to get objects and shape
    first_valid_file, objects, first_shape = _find_first_valid_npz(mask_npz_path, frame_files)
    if first_valid_file is None:
        print(f"  [FAIL] {seq_name}: no valid NPZ frame found")
        return False
    views, height, width = first_shape

    os.makedirs(dst_path, exist_ok=True)

    # Open shard writers for each object
    writers = {}
    for obj in objects:
        shard_path = join(dst_path, f"{obj}.shard")
        w = ShardWriter(shard_path, num_frames, compression_level)
        w.__enter__()
        writers[obj] = w

    # Pre-compute fallback compressed chunk for bad frames.
    # We write zero masks so index remains complete and conversion can continue.
    zero_mask = np.zeros((views, height, width), dtype=np.uint8)
    fallback_compressed = compress_mask_frame(zero_mask, compression_level)
    fallback_by_obj = {obj: fallback_compressed for obj in objects}

    bad_frames = []

    # Parallel compression with imap (ordered)
    t0 = time.time()
    worker_fn = partial(
        _compress_one_frame,
        compression_level=compression_level,
        expected_objects=objects,
        expected_shape=first_shape,
    )
    args_iter = [(f, mask_npz_path) for f in frame_files]
    frame_ids_written = []

    with Pool(processes=num_workers) as pool:
        for i, (frame_id, frame_file, compressed_dict, _shape, error_msg) in enumerate(
            pool.imap(worker_fn, args_iter, chunksize=32)
        ):
            if error_msg is not None:
                bad_frames.append({
                    "frame_id": frame_id,
                    "frame_file": frame_file,
                    "error": error_msg,
                })
                compressed_dict = fallback_by_obj

            # Write to each shard sequentially (main process)
            for obj in objects:
                writers[obj].write_frame_compressed(frame_id, compressed_dict[obj])
            frame_ids_written.append(frame_id)

            if (i + 1) % 500 == 0 or (i + 1) == num_frames:
                elapsed = time.time() - t0
                fps = (i + 1) / elapsed
                print(
                    f"    {i+1}/{num_frames} frames  "
                    f"({fps:.1f} frames/s, {elapsed:.0f}s, bad={len(bad_frames)})",
                    flush=True
                )

    # Close shard writers (backfills index)
    for w in writers.values():
        w.__exit__(None, None, None)

    # Write meta.json
    meta = {
        "version": 1,
        "codec": "lz4",
        "bitpacked": True,
        "objects": objects,
        "views": views,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "frame_ids": sorted(frame_ids_written),
        "bad_frame_ids": sorted([x["frame_id"] for x in bad_frames]),
        "bad_frame_count": len(bad_frames),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    if bad_frames:
        bad_path = join(dst_path, "bad_frames.json")
        with open(bad_path, "w") as f:
            json.dump(bad_frames, f, indent=2)
        print(f"  [WARN] {seq_name}: {len(bad_frames)} bad frames, see {bad_path}")

    elapsed = time.time() - t0
    # Report shard sizes
    total_shard_bytes = sum(
        os.path.getsize(join(dst_path, f"{obj}.shard")) for obj in objects
    )
    total_npz_bytes = sum(
        os.path.getsize(join(mask_npz_path, f)) for f in frame_files
    )
    print(
        f"  Done {seq_name}: {elapsed:.0f}s, "
        f"NPZ {total_npz_bytes/1e9:.2f}GB → Shard {total_shard_bytes/1e9:.2f}GB "
        f"({total_shard_bytes/total_npz_bytes:.2f}x)"
    )
    return True


def validate_sequence(seq_name, src_root, dst_root, num_samples=5):
    """Validate shard output matches NPZ for random frames."""
    mask_npz_path = join(src_root, "mask_npz", seq_name)
    shard_path = join(dst_root, "mask_shards", seq_name)
    meta_path = join(shard_path, "meta.json")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    objects = meta["objects"]
    views = meta["views"]
    height = meta["height"]
    width = meta["width"]
    frame_ids = meta["frame_ids"]
    bad_frame_ids = set(meta.get("bad_frame_ids", []))

    # Pick random frames
    rng = np.random.default_rng(42)
    valid_frame_ids = [fid for fid in frame_ids if fid not in bad_frame_ids]
    if len(valid_frame_ids) == 0:
        print(f"  [SKIP] {seq_name}: no valid frames for validation")
        return True
    sample_ids = rng.choice(
        valid_frame_ids, size=min(num_samples, len(valid_frame_ids)), replace=False
    )

    readers = {obj: ShardReader(join(shard_path, f"{obj}.shard")) for obj in objects}

    mismatches = 0
    for fid in sample_ids:
        # Read NPZ
        npz_file = f"{fid:06d}.npz"
        npz_data = np.load(join(mask_npz_path, npz_file))

        for obj in objects:
            # NPZ reference (auto-unpack if bitpacked)
            npz_mask = _load_mask_auto(npz_data, obj)  # (views, H, W) uint8

            # Shard
            shard_mask = readers[obj].read_frame(fid, views, height, width)

            # Compare: NPZ has {0, 255}, shard reconstructs to {0, 255}
            if not np.array_equal(npz_mask, shard_mask):
                # Check if the difference is just in non-binary values
                npz_binary = (npz_mask > 0).astype(np.uint8)
                shard_binary = (shard_mask > 0).astype(np.uint8)
                if np.array_equal(npz_binary, shard_binary):
                    pass  # Binary content matches (value difference only)
                else:
                    print(f"  [MISMATCH] {seq_name} frame {fid} object {obj}")
                    mismatches += 1

        npz_data.close()

    for r in readers.values():
        r.close()

    if mismatches == 0:
        print(f"  [VALID] {seq_name}: {len(sample_ids)} frames verified OK")
    else:
        print(f"  [FAIL] {seq_name}: {mismatches} mismatches in {len(sample_ids)} frames")
    return mismatches == 0


def main():
    parser = argparse.ArgumentParser(description="Convert masks from NPZ to LZ4 shard format")
    parser.add_argument("--src_root", required=True, help="Source HOI-M3 root (contains mask_npz/)")
    parser.add_argument("--dst_root", required=True, help="Destination root (will create mask_shards/)")
    parser.add_argument("--sequences", nargs="*", default=None,
                        help="Specific sequences to convert (default: all)")
    parser.add_argument("--num_workers", type=int, default=12,
                        help="Number of multiprocessing workers")
    parser.add_argument("--compression_level", type=int, default=9,
                        help="LZ4 compression level (1-16, higher=smaller)")
    parser.add_argument("--validate", action="store_true",
                        help="Validate converted data against source")
    parser.add_argument("--validate_samples", type=int, default=5,
                        help="Number of frames to validate per sequence")
    args = parser.parse_args()

    # Discover sequences
    npz_root = join(args.src_root, "mask_npz")
    if args.sequences:
        sequences = args.sequences
    else:
        sequences = sorted([
            d for d in os.listdir(npz_root)
            if isdir(join(npz_root, d)) and not d.endswith(".tar")
        ])

    print(f"Converting {len(sequences)} sequences")
    print(f"  Source: {npz_root}")
    print(f"  Destination: {join(args.dst_root, 'mask_shards')}")
    print(f"  Workers: {args.num_workers}, Compression: {args.compression_level}")
    print()

    t_total = time.time()
    success = 0
    for i, seq in enumerate(sequences):
        print(f"[{i+1}/{len(sequences)}] {seq}")
        if not isdir(join(npz_root, seq)):
            print(f"  [SKIP] not found")
            continue
        ok = convert_sequence(seq, args.src_root, args.dst_root,
                              args.num_workers, args.compression_level)
        if ok and args.validate:
            validate_sequence(seq, args.src_root, args.dst_root, args.validate_samples)
        if ok:
            success += 1
        print()

    elapsed = time.time() - t_total
    print(f"Done: {success}/{len(sequences)} sequences in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
