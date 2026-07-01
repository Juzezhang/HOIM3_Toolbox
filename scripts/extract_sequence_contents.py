#!/usr/bin/env python3
"""Extract mask contents (humans + objects) for every HOI-M3 sequence.

Data source priority:
  1. mask_npz  – NPZ keys are the most authoritative
  2. mocap     – object/ subdirs + smpl/ JSON array length
  3. M3.xlsx   – IMU column with "number objectname(angles)" format

Output: sequence_contents.json
"""

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = Path("/simurgh/group/juze/datasets/HOI-M3")
VIDEOS_DIR = BASE / "videos"
MASK_NPZ_DIR = BASE / "mask_npz"
MOCAP_DIR = BASE / "mocap"
SCANNED_OBJ_DIR = BASE / "scanned_object"
M3_XLSX = Path("/simurgh/u/juze/code/HOIM3_Toolbox/M3.xlsx")
OUTPUT_PATH = BASE / "sequence_contents.json"

# ── Name corrections for known NPZ typos ──────────────────────────────────
NAME_CORRECTIONS = {
    "bedsidecupboard": "bedside_cupboard",
    "cutlerytray": "cutlery_tray",
    "matermelon": "watermelon",
    "ffilebox": "filebox",
}

# IMU texts that are notes/status, not object lists
IMU_SKIP_PATTERNS = {
    "Yes", "No", "yes", "no", "ReDONE",
}
# Chinese note patterns that indicate unparseable IMU text
IMU_NOTE_RE = re.compile(r"[\u4e00-\u9fff]")  # contains Chinese characters

# Manual overrides for sequences where no automated source gives clean data.
# office_data18 (office_scene0206): IMU text is a Chinese note, no NPZ or mocap.
# Content inferred from same-scene neighbors office_data17/19/20.
MANUAL_OVERRIDES = {
    "office_data18": {
        "num_humans": 5,
        "humans": ["person0", "person1", "person2", "person3", "person4"],
        "num_objects": 7,
        "objects": ["bigsofa", "broom", "displayer", "keyboard", "laptop",
                     "officechair", "officedesk"],
        "source": "manual_override",
        "issues": ["inferred from same-scene neighbors office_data17/19/20"],
    },
}


def correct_name(name: str) -> str:
    return NAME_CORRECTIONS.get(name, name)


# ── Source 1: mask_npz ────────────────────────────────────────────────────
def read_npz_keys(seq: str) -> dict | None:
    """Read NPZ keys from the first frame. Returns dict or None on failure."""
    seq_dir = MASK_NPZ_DIR / seq
    if not seq_dir.is_dir():
        return None

    npz_files = sorted(seq_dir.glob("*.npz"))
    # Handle nested directory (e.g. mask_npz/seq/seq/*.npz)
    if not npz_files:
        nested = seq_dir / seq
        if nested.is_dir():
            npz_files = sorted(nested.glob("*.npz"))
    if not npz_files:
        return None

    data = np.load(npz_files[0])
    keys = [k for k in data.files if k != ".DS_Store"]

    persons = sorted([k for k in keys if k.startswith("person")])
    non_person = [k for k in keys if not k.startswith("person")]

    # All-numeric non-person keys → corrupted masks
    if non_person and all(k.isdigit() for k in non_person):
        return None

    # Filter out numeric keys from mixed sets (e.g. diningroom_data14)
    objects = sorted(set(correct_name(k) for k in non_person if not k.isdigit()))

    issues = []
    # Check for filtered numeric keys
    numeric_keys = [k for k in non_person if k.isdigit()]
    if numeric_keys:
        issues.append(f"filtered_numeric_keys: {numeric_keys}")

    # Check for name corrections applied
    corrected = {k: correct_name(k) for k in non_person if k in NAME_CORRECTIONS}
    if corrected:
        issues.append(f"name_corrections: {corrected}")

    if ".DS_Store" in data.files:
        issues.append("filtered_.DS_Store")

    return {
        "num_humans": len(persons),
        "humans": persons,
        "num_objects": len(objects),
        "objects": objects,
        "source": "mask_npz",
        "issues": issues if issues else None,
    }


# ── Source 2: mocap ───────────────────────────────────────────────────────
def read_mocap(seq: str) -> dict | None:
    """Read object names from mocap/object/ and human count from mocap/smpl/."""
    mocap_seq = MOCAP_DIR / seq
    if not mocap_seq.is_dir():
        return None

    # Objects from subdirectories
    obj_dir = mocap_seq / "object"
    if obj_dir.is_dir():
        objects = sorted(
            d.name for d in obj_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )
    else:
        objects = []

    # Human count from SMPL JSON array length
    smpl_dir = mocap_seq / "smpl"
    num_humans = 0
    if smpl_dir.is_dir():
        smpl_files = sorted(smpl_dir.glob("*.json"))
        if smpl_files:
            with open(smpl_files[0]) as f:
                smpl_data = json.load(f)
            if isinstance(smpl_data, list):
                num_humans = len(smpl_data)

    if not objects and num_humans == 0:
        return None

    humans = [f"person{i}" for i in range(num_humans)]
    return {
        "num_humans": num_humans,
        "humans": humans,
        "num_objects": len(objects),
        "objects": objects,
        "source": "mocap",
        "issues": None,
    }


# ── Source 3: M3.xlsx IMU column ──────────────────────────────────────────
def load_m3_imu_map() -> dict[str, str]:
    """Load Newname → IMU column mapping from M3.xlsx."""
    df = pd.read_excel(M3_XLSX)
    mapping = {}
    for _, row in df.iterrows():
        newname = row.get("Newname")
        imu = row.get("IMU数据是否正常")
        if pd.notna(newname) and pd.notna(imu):
            newname = str(newname).strip()
            imu = str(imu).strip()
            if newname and newname != "Newname":
                mapping[newname] = imu
    return mapping


def parse_imu_objects(imu_text: str) -> dict | None:
    """Parse IMU column text in format 'number objectname [angles]' per line.

    Returns dict with objects and inferred num_humans, or None if unparseable.
    """
    if not imu_text or imu_text in IMU_SKIP_PATTERNS:
        return None

    lines = imu_text.strip().split("\n")
    objects = []
    first_obj_num = None

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Match: number objectname [optional angles/rotation info]
        # Formats: "3 bed", "3 bed(0,90,0)", "3 bed 90 0 270", "3 bed- 0 90 0"
        m = re.match(
            r"^(\d+)\s+"           # leading number
            r"([a-zA-Z_]+)",       # object name (letters and underscores)
            line,
        )
        if m:
            num = int(m.group(1))
            name = m.group(2).rstrip("-")  # strip trailing dash
            if first_obj_num is None:
                first_obj_num = num
            objects.append(correct_name(name))
        # Also try: "objectname(angles)" without leading number
        elif line[0].isascii() and line[0].isalpha():
            m2 = re.match(r"^([a-zA-Z_]+)", line)
            if m2:
                name = m2.group(1)
                # Skip short words and status tokens
                if len(name) > 2 and name not in IMU_SKIP_PATTERNS:
                    objects.append(correct_name(name))

    if not objects:
        return None

    objects = sorted(set(objects))

    # Infer num_humans from the first object number
    # Convention: persons are numbered 1..N, objects start at N+1
    num_humans = (first_obj_num - 1) if first_obj_num is not None else 0
    humans = [f"person{i}" for i in range(num_humans)]

    return {
        "num_humans": num_humans,
        "humans": humans,
        "num_objects": len(objects),
        "objects": objects,
        "source": "m3_xlsx_imu",
        "issues": None,
    }


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    sequences = sorted(
        d.name for d in VIDEOS_DIR.iterdir() if d.is_dir()
    )
    print(f"Total sequences: {len(sequences)}")

    scanned_objects = set(
        d.name for d in SCANNED_OBJ_DIR.iterdir() if d.is_dir()
    )

    imu_map = load_m3_imu_map()

    results = {}
    stats = {"mask_npz": 0, "mocap": 0, "m3_xlsx_imu": 0, "manual_override": 0, "missing": 0}
    validation_warnings = []

    for seq in sequences:
        # Check manual overrides first
        if seq in MANUAL_OVERRIDES:
            entry = dict(MANUAL_OVERRIDES[seq])  # copy
        else:
            # Try sources in priority order
            entry = read_npz_keys(seq)
            if entry is None:
                entry = read_mocap(seq)
            if entry is None:
                imu_text = imu_map.get(seq)
                if imu_text:
                    entry = parse_imu_objects(imu_text)

        if entry is None:
            print(f"  WARNING: no data for {seq}")
            stats["missing"] += 1
            entry = {
                "num_humans": 0,
                "humans": [],
                "num_objects": 0,
                "objects": [],
                "source": "missing",
                "issues": ["no data source available"],
            }
        else:
            stats[entry["source"]] += 1

        # Validate objects against scanned_object/
        missing_meshes = [
            obj for obj in entry["objects"] if obj not in scanned_objects
        ]
        if missing_meshes:
            entry.setdefault("issues", None)
            if entry["issues"] is None:
                entry["issues"] = []
            entry["issues"].append(f"no_mesh: {missing_meshes}")
            validation_warnings.append((seq, missing_meshes))

        results[seq] = entry

    # Write output (clean version without source/issues)
    clean_results = {}
    for seq, entry in results.items():
        clean_results[seq] = {
            "num_humans": entry["num_humans"],
            "humans": entry["humans"],
            "num_objects": entry["num_objects"],
            "objects": entry["objects"],
        }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(clean_results, f, indent=2)

    # Write detailed version with source/issues for debugging
    debug_path = OUTPUT_PATH.with_name("sequence_contents_debug.json")
    with open(debug_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\nSource breakdown:")
    for src, count in stats.items():
        print(f"  {src}: {count}")
    print(f"\nTotal: {sum(stats.values())}")

    if validation_warnings:
        print(f"\nObjects without scanned mesh ({len(validation_warnings)} sequences):")
        for seq, missing in validation_warnings[:10]:
            print(f"  {seq}: {missing}")
        if len(validation_warnings) > 10:
            print(f"  ... and {len(validation_warnings) - 10} more")

    print(f"\nOutput: {OUTPUT_PATH}")
    print(f"Debug:  {debug_path}")


if __name__ == "__main__":
    main()
