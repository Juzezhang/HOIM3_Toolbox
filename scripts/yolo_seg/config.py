"""Shared constants and class mapping for the YOLO-seg + matching pipeline."""

import json
import os
from os.path import join

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT = "/simurgh/group/juze/datasets/HOI-M3"
SHARD_ROOT = "/simurgh2/datasets/HOI-M3/mask_shards"
VALIDITY_ROOT = join(DATA_ROOT, "mask_validity")
VIDEO_ROOT = join(DATA_ROOT, "videos")
SEQ_CONTENTS = join(DATA_ROOT, "sequence_contents.json")
WDS_ROOT = "/simurgh2/datasets/HOI-M3/yolo_seg_wds"

# ── Data constants ────────────────────────────────────────────────────────────
MASK_H, MASK_W = 1080, 1920
NUM_VIEWS = 42
FRAME_STEP = 60          # 1 fps subsample
IMGSZ = 640              # YOLO input width (height = 360 for 16:9)
SAMPLES_PER_SHARD = 5000

# ── Name corrections (from extract_sequence_contents.py) ─────────────────────
NAME_CORRECTIONS = {
    "bedsidecupboard": "bedside_cupboard",
    "cutlerytray": "cutlery_tray",
    "matermelon": "watermelon",
    "ffilebox": "filebox",
}


def correct_name(name: str) -> str:
    """Apply known name corrections to object keys."""
    return NAME_CORRECTIONS.get(name, name)


def load_sequence_contents() -> dict:
    """Load sequence_contents.json → {seq_name: {num_humans, humans, objects, ...}}."""
    with open(SEQ_CONTENTS) as f:
        return json.load(f)


def build_class_mapping(seq_contents: dict) -> tuple:
    """Build class mapping: person → 0, sorted objects → 1..N.

    Returns (class_to_id, class_names) where class_names[i] is the name for id i.
    """
    all_objects = set()
    for v in seq_contents.values():
        all_objects.update(v["objects"])
    sorted_objects = sorted(all_objects)

    class_to_id = {"person": 0}
    class_names = ["person"]
    for i, obj in enumerate(sorted_objects, 1):
        class_to_id[obj] = i
        class_names.append(obj)

    return class_to_id, class_names


def get_train_sequences(seq_contents: dict) -> list:
    """Return sequences that have mask shards (training set)."""
    return sorted(
        s for s in seq_contents
        if os.path.isdir(join(SHARD_ROOT, s))
    )


def get_test_sequences(seq_contents: dict) -> list:
    """Return sequences without mask shards (test / inference only).

    After fixing the nested mask_npz directory issue (36 sequences had NPZ files
    at mask_npz/{seq}/{seq}/ instead of mask_npz/{seq}/), the split is:
      - 190 training sequences (have mask shards)
      - 14 test sequences (no masks at all)
    All 87 object classes are covered in the training set.
    """
    return sorted(
        s for s in seq_contents
        if not os.path.isdir(join(SHARD_ROOT, s))
    )
