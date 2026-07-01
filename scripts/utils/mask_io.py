"""
Mask I/O utilities supporting NPZ (zlib) and object-major shard (LZ4+bitpack) formats.

Shard format stores one file per object per sequence, with an embedded frame index.
Each frame's mask is bit-packed (1 bit/pixel) then LZ4-compressed for fast parallel reads.

File layout (.shard):
    [Header 10B] [Index num_frames×16B] [Frame 0 data] [Frame 1 data] ...
"""
import os
import json
import struct
import numpy as np
import lz4.frame
from os.path import join
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

# Shard binary format constants
_MAGIC = b"MSK1"
_HEADER_FORMAT = "<4sHI"  # magic(4B) version(u16) num_frames(u32)
_HEADER_SIZE = struct.calcsize(_HEADER_FORMAT)  # 10 bytes
_INDEX_ENTRY_FORMAT = "<IQI"  # frame_id(u32) offset(u64) comp_size(u32)
_INDEX_ENTRY_SIZE = struct.calcsize(_INDEX_ENTRY_FORMAT)  # 16 bytes


class ShardWriter:
    """Writes a single object shard file.

    Usage:
        with ShardWriter(path, num_frames) as w:
            for frame_id, mask_array in frames:
                w.write_frame(frame_id, mask_array)
    """

    def __init__(self, path: str, num_frames: int, compression_level: int = 9):
        self.path = path
        self.num_frames = num_frames
        self.compression_level = compression_level
        self.index_entries: List[Tuple[int, int, int]] = []
        self.f = None

    def __enter__(self):
        self.f = open(self.path, "wb")
        # Write header
        self.f.write(struct.pack(_HEADER_FORMAT, _MAGIC, 1, self.num_frames))
        # Reserve index space (will overwrite later)
        self._index_offset = self.f.tell()
        self.f.write(b"\x00" * (self.num_frames * _INDEX_ENTRY_SIZE))
        return self

    def write_frame_compressed(self, frame_id: int, compressed: bytes):
        """Write a pre-compressed frame chunk."""
        offset = self.f.tell()
        self.f.write(compressed)
        self.index_entries.append((frame_id, offset, len(compressed)))

    def __exit__(self, *exc):
        if self.f is None:
            return
        # Write index back to reserved space
        self.f.seek(self._index_offset)
        for frame_id, offset, comp_size in self.index_entries:
            self.f.write(struct.pack(_INDEX_ENTRY_FORMAT, frame_id, offset, comp_size))
        self.f.close()
        self.f = None


class ShardReader:
    """Reads frames from an object shard file.

    Keeps the file handle open for fast repeated access.
    Thread-safe for read_frame() calls from different threads
    ONLY if each thread accesses different ShardReader instances.
    """

    def __init__(self, shard_path: str):
        self.shard_path = shard_path
        self.f = open(shard_path, "rb")
        self._read_header_and_index()

    def _read_header_and_index(self):
        raw = self.f.read(_HEADER_SIZE)
        magic, version, num_frames = struct.unpack(_HEADER_FORMAT, raw)
        if magic != _MAGIC:
            raise ValueError(f"Invalid shard magic: {magic}")
        self.version = version
        self.num_frames = num_frames

        # Read full index
        index_bytes = self.f.read(num_frames * _INDEX_ENTRY_SIZE)
        self.frame_index: Dict[int, Tuple[int, int]] = {}
        for i in range(num_frames):
            start = i * _INDEX_ENTRY_SIZE
            frame_id, offset, comp_size = struct.unpack(
                _INDEX_ENTRY_FORMAT, index_bytes[start : start + _INDEX_ENTRY_SIZE]
            )
            self.frame_index[frame_id] = (offset, comp_size)

    def read_frame(self, frame_id: int, views: int, height: int, width: int) -> np.ndarray:
        """Read and decompress a single frame, returning full (views, H, W) uint8 {0,255}."""
        offset, comp_size = self.frame_index[frame_id]
        self.f.seek(offset)
        compressed = self.f.read(comp_size)
        raw = lz4.frame.decompress(compressed)
        packed_width = (width + 7) // 8
        packed = np.frombuffer(raw, dtype=np.uint8).reshape(views, height, packed_width)
        full = np.unpackbits(packed, axis=-1)[:, :, :width]
        full *= 255
        return full  # (views, H, W) uint8

    def read_frame_views(
        self, frame_id: int, view_indices: List[int],
        views: int, height: int, width: int
    ) -> Dict[int, np.ndarray]:
        """Read a frame and return only selected views as a dict."""
        full = self.read_frame(frame_id, views, height, width)
        return {v: full[v].copy() for v in view_indices if v < views}

    @property
    def frame_ids(self) -> List[int]:
        return sorted(self.frame_index.keys())

    def close(self):
        if self.f is not None:
            self.f.close()
            self.f = None

    def __del__(self):
        self.close()


def compress_mask_frame(mask: np.ndarray, compression_level: int = 9) -> bytes:
    """Compress a single mask array (views, H, W) uint8 → LZ4(bitpacked bytes)."""
    packed = np.packbits(mask > 0, axis=-1)  # (views, H, ceil(W/8))
    return lz4.frame.compress(
        packed.tobytes(),
        compression_level=compression_level,
    )


# --- Sequence-level shard loading ---

class SequenceShardReaders:
    """Manages ShardReaders for all objects in a sequence. Reusable across batches."""

    def __init__(self, seq_path: str):
        self.seq_path = seq_path
        meta_path = join(seq_path, "meta.json")
        with open(meta_path, "r") as f:
            self.meta = json.load(f)
        self.objects = self.meta["objects"]
        self.views = self.meta["views"]
        self.height = self.meta["height"]
        self.width = self.meta["width"]
        self.frame_ids_list = self.meta["frame_ids"]

        self.readers: Dict[str, ShardReader] = {}
        for obj in self.objects:
            shard_path = join(seq_path, f"{obj}.shard")
            self.readers[obj] = ShardReader(shard_path)

    def close(self):
        for r in self.readers.values():
            r.close()
        self.readers.clear()

    def __del__(self):
        self.close()


def _read_object_frame(reader: ShardReader, frame_id: int,
                        view_indices: List[int], views: int,
                        height: int, width: int) -> Dict[int, np.ndarray]:
    """Worker function for thread-parallel per-object reads."""
    return reader.read_frame_views(frame_id, view_indices, views, height, width)


def _read_object_frame_full(reader: ShardReader, frame_id: int,
                             views: int, height: int, width: int) -> np.ndarray:
    """Worker: decompress full (V, H, W) array (no per-view copy)."""
    return reader.read_frame(frame_id, views, height, width)


def load_frame_masks_shard(
    seq_readers: SequenceShardReaders,
    frame_id: int,
    view_indices: List[int],
) -> Dict[str, Dict[int, np.ndarray]]:
    """Load one frame's masks from shards, decompressing objects in parallel threads."""
    v = seq_readers.views
    h = seq_readers.height
    w = seq_readers.width

    # Parallel per-object reads (LZ4 releases GIL)
    frame_data = {}
    with ThreadPoolExecutor(max_workers=min(len(seq_readers.objects), 7)) as pool:
        futures = {
            obj: pool.submit(_read_object_frame, reader, frame_id, view_indices, v, h, w)
            for obj, reader in seq_readers.readers.items()
        }
        for obj, fut in futures.items():
            frame_data[obj] = fut.result()
    return frame_data


def load_frame_masks_shard_full(
    seq_readers: SequenceShardReaders,
    frame_id: int,
) -> Dict[str, np.ndarray]:
    """Load one frame's masks as full (V, H, W) arrays — no per-view copy.

    Returns {obj_name: ndarray(V, H, W) uint8}. Each array is contiguous
    and suitable for stacking across frames for GPU upload.
    """
    v = seq_readers.views
    h = seq_readers.height
    w = seq_readers.width

    frame_data = {}
    with ThreadPoolExecutor(max_workers=min(len(seq_readers.objects), 7)) as pool:
        futures = {
            obj: pool.submit(_read_object_frame_full, reader, frame_id, v, h, w)
            for obj, reader in seq_readers.readers.items()
        }
        for obj, fut in futures.items():
            frame_data[obj] = fut.result()
    return frame_data


# --- NPZ loading (existing logic) ---

def load_frame_masks_npz(
    mask_npz_path: str, frame_file: str, view_indices: List[int]
) -> Dict[str, Dict[int, np.ndarray]]:
    """Load masks from NPZ format."""
    data = np.load(join(mask_npz_path, frame_file))
    frame_data = {}
    for key in data.keys():
        all_masks = data[key]  # (42, H, W) uint8
        masks = {}
        for v in view_indices:
            if v < all_masks.shape[0]:
                masks[v] = all_masks[v].copy()
        frame_data[key] = masks
    data.close()
    return frame_data


# --- Format detection ---

def detect_mask_format(root_path: str, seq_name: str, mask_root: Optional[str] = None) -> str:
    """Auto-detect mask format for a sequence. Returns 'shard' or 'npz'."""
    if mask_root:
        shard_path = join(mask_root, seq_name, "meta.json")
    else:
        shard_path = join(root_path, "mask_shards", seq_name, "meta.json")
    if os.path.isfile(shard_path):
        return "shard"
    return "npz"
