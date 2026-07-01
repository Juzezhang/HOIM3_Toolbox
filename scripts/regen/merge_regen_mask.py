#!/usr/bin/env python3
"""Merge-regen mask repair: selectively replace invalid (frame, view, person) cells.

Pipeline per sequence:
  1. (Optional) Run YOLO+SAM inference for the whole sequence into a temp dir,
     producing 720p masks per (frame, view) with named keys
     {personK, obj_name} — same structure as mask_npz_generated.
     If --reuse_npz_generated is set, we reuse the existing
     /simurgh2/datasets/HOI-M3/mask_npz_generated/<seq>/ outputs directly.
  2. For each frame: load original 1080p shards + validity npz. For each
     (object, view) cell:
       - validity == 1: keep ORIGINAL pixels untouched
       - validity == 0: write FRESH YOLO+SAM pixels (upscaled 720→1080 NEAREST)
         For person* cells, perform per-frame ReID alignment to map fresh
         person ids ↔ canonical person ids using anchor embeddings derived
         from the valid-view crops of the SAME frame.
  3. Atomically rewrite each <obj>.shard via <obj>.shard.new + os.replace,
     preserving the existing 1080p meta.json (no re-upgrade needed).
  4. Optionally re-run multi-view voxel-consistency validity check to refresh
     the validity npz files (delegated to multi_view_mask_check.py as subprocess).

Notes:
  - Bit-identical preservation of validity=1 cells is guaranteed because the
    original (V, H, W) buffer is read once and only invalid view-slices are
    overwritten before re-packing.
  - .bak copies of the original shards are created before atomic rename
    (cheap reflink/hardlink not used to keep portability across simurgh*).
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from os.path import join
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, '/simurgh/u/juze/code/HOIM3_Toolbox')
from scripts.utils.mask_io import (  # noqa: E402
    ShardReader,
    ShardWriter,
    compress_mask_frame,
)


DEFAULT_SHARD_ROOT = "/simurgh2/datasets/HOI-M3/mask_shards"
DEFAULT_VALIDITY_ROOTS = [
    "/scr/juze/datasets/HOI-M3/mask_validity",
    "/simurgh2/datasets/HOI-M3/mask_validity",
    "/simurgh/group/juze/datasets/HOI-M3/mask_validity",
]
DEFAULT_GENERATED_ROOT = "/simurgh2/datasets/HOI-M3/mask_npz_generated"
DEFAULT_IMAGE_ROOT = "/simurgh2/datasets/HOI-M3/images"
DEFAULT_REID_MODEL = "/simurgh2/datasets/HOI-M3/reid_data/runs/best_reid.pt"

NUM_VIEWS = 42
SHARD_H, SHARD_W = 1080, 1920
SRC_H, SRC_W = 720, 1280  # NPZ-generated source resolution


# ── Validity resolution ──────────────────────────────────────────────────────

def resolve_validity_root(seq: str, override: Optional[str]) -> str:
    if override:
        d = join(override, seq)
        if os.path.isdir(d):
            return d
        raise FileNotFoundError(f"validity_root override has no {seq}: {override}")
    for root in DEFAULT_VALIDITY_ROOTS:
        d = join(root, seq)
        if os.path.isdir(d):
            return d
    raise FileNotFoundError(
        f"No mask_validity/{seq} found in any of {DEFAULT_VALIDITY_ROOTS}"
    )


def load_validity_frame(validity_dir: str, frame_id: int, obj_names: List[str]) -> Dict[str, np.ndarray]:
    """Return {obj: (V,) uint8 array} for one frame."""
    path = join(validity_dir, f"{frame_id:06d}.npz")
    if not os.path.exists(path):
        # Missing validity → treat as all-invalid? Safer: treat as all-valid so we DON'T
        # overwrite. But we want to refresh missing frames. Use all-valid (skip) to be safe.
        return {obj: np.ones(NUM_VIEWS, dtype=np.uint8) for obj in obj_names}
    d = np.load(path)
    out = {}
    for obj in obj_names:
        key = f"{obj}_validity"
        if key in d.files:
            out[obj] = d[key].astype(np.uint8)
        else:
            out[obj] = np.ones(NUM_VIEWS, dtype=np.uint8)
    d.close()
    return out


# ── Fresh mask loading from mask_npz_generated ───────────────────────────────

def load_generated_frame(gen_dir: str, frame_id: int) -> Optional[Dict[str, np.ndarray]]:
    """Load fresh YOLO+SAM masks for a frame, or None if missing.

    Returns {obj_or_person_name: (V, SRC_H, SRC_W) uint8 {0,255}}.
    """
    p = join(gen_dir, f"{frame_id:06d}.npz")
    if not os.path.exists(p):
        return None
    d = np.load(p)
    out = {k: d[k] for k in d.files}
    # numpy lazy → realize via copy to release file handle
    out = {k: np.ascontiguousarray(v) for k, v in out.items()}
    d.close()
    return out


# ── 720 → 1080 upscaling ─────────────────────────────────────────────────────

def upscale_view(mask_720: np.ndarray) -> np.ndarray:
    """(H720, W1280) uint8 → (H1080, W1920) uint8 via NEAREST."""
    if mask_720.shape == (SHARD_H, SHARD_W):
        return mask_720
    return cv2.resize(mask_720, (SHARD_W, SHARD_H), interpolation=cv2.INTER_NEAREST)


# ── ReID alignment ───────────────────────────────────────────────────────────

class ReIDAligner:
    """Per-frame ReID alignment of fresh person masks to canonical person IDs.

    For each frame we:
      - extract anchor embeddings from VALID-view original masks of each
        canonical personK (using the raw view image cropped by mask bbox)
      - extract candidate embeddings from FRESH person masks at INVALID views
      - assign each invalid (view, candidate) to the canonical personK with
        highest cosine similarity (above threshold). Ties broken by view-index.
    """

    def __init__(self, reid_model_path: str, device: str = "cuda",
                 sim_threshold: float = 0.5):
        from scripts.yolo_seg.train_reid import ReIDModel
        import torch
        self.torch = torch
        self.device = torch.device(device)
        self.sim_threshold = sim_threshold

        ckpt = torch.load(reid_model_path, map_location=self.device, weights_only=False)
        self.model = ReIDModel(num_classes=441, embed_dim=256).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self._mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def _embed_crops(self, crops_rgb: List[np.ndarray]) -> np.ndarray:
        """List of variable-size RGB uint8 arrays → (N, 256) cosine-normalized embeddings."""
        if not crops_rgb:
            return np.zeros((0, 256), dtype=np.float32)
        arr = np.zeros((len(crops_rgb), 3, 256, 128), dtype=np.float32)
        for i, c in enumerate(crops_rgb):
            if c.size == 0:
                continue
            resized = cv2.resize(c, (128, 256)).transpose(2, 0, 1).astype(np.float32) / 255.0
            arr[i] = resized
        t = self.torch.from_numpy(arr).to(self.device)
        t = (t - self._mean) / self._std
        with self.torch.no_grad():
            emb, _ = self.model(t)
        return emb.cpu().numpy()

    @staticmethod
    def _crop_from_mask(image_bgr: np.ndarray, mask: np.ndarray,
                        pad: int = 8) -> Optional[np.ndarray]:
        """Tight crop around mask, returned as RGB. None if mask empty."""
        ys, xs = np.where(mask > 0)
        if ys.size == 0:
            return None
        y1, y2 = max(0, ys.min() - pad), min(mask.shape[0], ys.max() + pad + 1)
        x1, x2 = max(0, xs.min() - pad), min(mask.shape[1], xs.max() + pad + 1)
        crop_bgr = image_bgr[y1:y2, x1:x2]
        if crop_bgr.size == 0:
            return None
        return cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

    def align_frame(
        self,
        person_keys: List[str],
        original_person_masks: Dict[str, np.ndarray],  # personK -> (V, H, W) uint8 (1080p)
        fresh_person_masks: Dict[str, np.ndarray],     # personK -> (V, h, w) uint8 (720p, from inference)
        validity: Dict[str, np.ndarray],                # personK -> (V,) uint8
        load_image_view: callable,                      # v -> BGR np.ndarray or None at 720p
    ) -> Dict[str, Dict[int, str]]:
        """Decide, for each canonical personK and each INVALID view v, which
        fresh personJ's mask to write (or None to leave empty).

        Returns: {canonical_pk: {view_v: fresh_pk_or_None}}.

        Strategy:
          1. For each canonical pk, gather crops from its VALID views using
             original 1080p mask (downscaled bbox to 720p image space).
             Average their embeddings → anchor[pk].
          2. For each INVALID view v, for each fresh person key pj with a
             non-empty fresh mask at view v, compute its embedding from a
             crop of the 720p image at that view.
          3. Cosine-similarity match: assign each fresh embedding to the best
             canonical anchor (above threshold). For a given canonical pk +
             invalid view v, prefer the fresh pj whose embedding best matches
             anchor[pk] AND whose best anchor IS pk (mutual best). Ties → pj
             with higher mask area.
        """
        # Step 1: anchors from valid views
        anchor_crops: Dict[str, List[np.ndarray]] = {pk: [] for pk in person_keys}
        for pk in person_keys:
            vmask = original_person_masks.get(pk)
            if vmask is None:
                continue
            valid_views = np.where(validity.get(pk, np.zeros(NUM_VIEWS, np.uint8)) == 1)[0]
            # Limit to <=6 valid views for speed
            for v in valid_views[:6]:
                img = load_image_view(int(v))
                if img is None:
                    continue
                # Downscale original 1080p mask to 720p to match raw image
                m_720 = cv2.resize(vmask[v], (SRC_W, SRC_H),
                                   interpolation=cv2.INTER_NEAREST)
                if m_720.sum() < 500:  # skip tiny anchors
                    continue
                crop = self._crop_from_mask(img, m_720)
                if crop is not None:
                    anchor_crops[pk].append(crop)

        anchor_embs: Dict[str, Optional[np.ndarray]] = {}
        # Flatten for one batched forward
        flat_crops = []
        flat_owner = []  # list of pk for each crop
        for pk in person_keys:
            for c in anchor_crops[pk]:
                flat_crops.append(c)
                flat_owner.append(pk)
        if flat_crops:
            all_embs = self._embed_crops(flat_crops)
            for pk in person_keys:
                idxs = [i for i, o in enumerate(flat_owner) if o == pk]
                if idxs:
                    a = all_embs[idxs].mean(axis=0)
                    n = np.linalg.norm(a) + 1e-9
                    anchor_embs[pk] = a / n
                else:
                    anchor_embs[pk] = None
        else:
            for pk in person_keys:
                anchor_embs[pk] = None

        # Step 2: per-view candidate crops from FRESH masks (720p)
        assignment: Dict[str, Dict[int, Optional[str]]] = {pk: {} for pk in person_keys}

        # Collect all invalid (view, fresh_pj) candidates
        candidates_by_view: Dict[int, List[Tuple[str, np.ndarray]]] = defaultdict(list)
        # Union of views that are invalid for ANY canonical pk
        union_invalid_views: set = set()
        for pk in person_keys:
            inv = np.where(validity.get(pk, np.zeros(NUM_VIEWS, np.uint8)) == 0)[0]
            for v in inv:
                union_invalid_views.add(int(v))

        # For each invalid view, get image once, then list candidate fresh persons
        view_to_image: Dict[int, np.ndarray] = {}
        for v in union_invalid_views:
            img = load_image_view(v)
            if img is None:
                continue
            view_to_image[v] = img
            for pj, fm in fresh_person_masks.items():
                if fm is None or v >= fm.shape[0]:
                    continue
                m = fm[v]
                if m.sum() < 500:
                    continue
                crop = self._crop_from_mask(img, m)
                if crop is not None:
                    candidates_by_view[v].append((pj, m))

        # Step 3: embed all candidate crops in one batch
        cand_flat: List[np.ndarray] = []
        cand_index: List[Tuple[int, str, np.ndarray]] = []  # (view, pj, mask_720)
        for v, lst in candidates_by_view.items():
            img = view_to_image[v]
            for pj, m in lst:
                crop = self._crop_from_mask(img, m)
                if crop is None:
                    continue
                cand_flat.append(crop)
                cand_index.append((v, pj, m))

        cand_embs = self._embed_crops(cand_flat) if cand_flat else np.zeros((0, 256), np.float32)

        # For each canonical pk and each of its invalid views, pick the best
        # candidate at that view by cosine similarity to anchor_embs[pk].
        # Mutual-best rule: candidate must also have its top anchor == pk.
        # Build per-(view) similarity matrix on the fly.
        by_view_cands: Dict[int, List[int]] = defaultdict(list)  # view -> [cand_idx]
        for ci, (v, pj, _) in enumerate(cand_index):
            by_view_cands[v].append(ci)

        for pk in person_keys:
            anchor = anchor_embs.get(pk)
            inv = np.where(validity.get(pk, np.zeros(NUM_VIEWS, np.uint8)) == 0)[0]
            for v in inv:
                v = int(v)
                cands_at_v = by_view_cands.get(v, [])
                if not cands_at_v or anchor is None:
                    assignment[pk][v] = None
                    continue
                best_ci = -1
                best_sim = -2.0
                for ci in cands_at_v:
                    e = cand_embs[ci]
                    n = np.linalg.norm(e) + 1e-9
                    sim = float(np.dot(anchor, e / n))
                    if sim > best_sim:
                        best_sim = sim
                        best_ci = ci
                if best_ci < 0 or best_sim < self.sim_threshold:
                    assignment[pk][v] = None
                    continue
                # Mutual-best check: cand_embs[best_ci] must prefer pk among anchors
                e = cand_embs[best_ci]; e = e / (np.linalg.norm(e) + 1e-9)
                top_pk = pk; top_sim = best_sim
                for opk in person_keys:
                    if opk == pk or anchor_embs.get(opk) is None:
                        continue
                    s = float(np.dot(anchor_embs[opk], e))
                    if s > top_sim:
                        top_sim = s; top_pk = opk
                if top_pk != pk:
                    assignment[pk][v] = None  # would belong to other canonical
                else:
                    assignment[pk][v] = cand_index[best_ci][1]
        return assignment


# ── Core merge logic ─────────────────────────────────────────────────────────

def merge_frame(
    frame_id: int,
    objects: List[str],
    person_keys: List[str],
    readers: Dict[str, ShardReader],
    generated_dir: str,
    image_dir: str,
    validity_dir: str,
    aligner: Optional[ReIDAligner],
) -> Dict[str, np.ndarray]:
    """Build merged (V, H, W) uint8 {0,255} mask for each object key.

    Returns {obj_or_person_key: (V, SHARD_H, SHARD_W) uint8}.
    """
    # 1. Read originals (1080p) — these are the default
    merged: Dict[str, np.ndarray] = {}
    for obj in objects:
        merged[obj] = readers[obj].read_frame(frame_id, NUM_VIEWS, SHARD_H, SHARD_W).copy()

    # 2. Validity per object
    validity = load_validity_frame(validity_dir, frame_id, objects)

    # 3. Fresh masks (720p) — may be None if inference missing
    fresh = load_generated_frame(generated_dir, frame_id)

    if fresh is None:
        # No fresh data: leave originals untouched (treat all as "kept")
        return merged

    # 4. Object masks: simple per-view replace at invalid cells (no ID alignment)
    obj_only = [o for o in objects if o not in person_keys]
    for obj in obj_only:
        if obj not in fresh:
            continue
        fmask = fresh[obj]  # (V, 720, 1280)
        v_arr = validity.get(obj, np.ones(NUM_VIEWS, np.uint8))
        for v in range(NUM_VIEWS):
            if v_arr[v] == 0 and v < fmask.shape[0]:
                merged[obj][v] = upscale_view(fmask[v])

    # 5. Person masks: need ReID alignment
    if person_keys and aligner is not None:
        # Build a cheap image-loader closure (cache views as needed)
        img_cache: Dict[int, Optional[np.ndarray]] = {}

        def load_image_view(v: int) -> Optional[np.ndarray]:
            if v not in img_cache:
                p = join(image_dir, str(v), f"{frame_id:06d}.jpg")
                img_cache[v] = cv2.imread(p)
            return img_cache[v]

        original_person_masks = {pk: merged[pk] for pk in person_keys if pk in merged}
        fresh_person_masks = {pk: fresh.get(pk) for pk in person_keys}

        assign = aligner.align_frame(
            person_keys=person_keys,
            original_person_masks=original_person_masks,
            fresh_person_masks=fresh_person_masks,
            validity={pk: validity.get(pk, np.ones(NUM_VIEWS, np.uint8))
                      for pk in person_keys},
            load_image_view=load_image_view,
        )

        for pk in person_keys:
            v_arr = validity.get(pk, np.ones(NUM_VIEWS, np.uint8))
            for v in range(NUM_VIEWS):
                if v_arr[v] == 1:
                    continue  # keep original
                source_pj = assign.get(pk, {}).get(v)
                if source_pj is None:
                    # Leave empty (validity is 0 anyway — was already
                    # considered bad; clearing is safer than keeping wrong id).
                    merged[pk][v] = 0
                else:
                    fmask = fresh.get(source_pj)
                    if fmask is None or v >= fmask.shape[0]:
                        merged[pk][v] = 0
                    else:
                        merged[pk][v] = upscale_view(fmask[v])
    else:
        # No aligner: just clear invalid person cells (do NOT trust raw YOLO ids)
        for pk in person_keys:
            v_arr = validity.get(pk, np.ones(NUM_VIEWS, np.uint8))
            for v in range(NUM_VIEWS):
                if v_arr[v] == 0:
                    merged[pk][v] = 0
    return merged


# ── Shard rewrite ────────────────────────────────────────────────────────────

def rewrite_sequence(
    seq: str,
    seq_dir: str,
    objects: List[str],
    frame_ids: List[int],
    generated_dir: str,
    image_dir: str,
    validity_dir: str,
    aligner: Optional[ReIDAligner],
    output_dir: Optional[str],
    keep_bak: bool = True,
    progress_every: int = 250,
) -> None:
    """Run merge for all frames and atomically rewrite shards."""
    person_keys = [o for o in objects if o.startswith("person")]

    # Output dir (in-place if not provided)
    out_dir = output_dir if output_dir else seq_dir
    os.makedirs(out_dir, exist_ok=True)

    # Open readers (one per object)
    readers = {obj: ShardReader(join(seq_dir, f"{obj}.shard")) for obj in objects}

    # Open writers to .new
    new_paths = {obj: join(out_dir, f"{obj}.shard.new") for obj in objects}
    writers = {obj: ShardWriter(new_paths[obj], len(frame_ids), compression_level=6)
               for obj in objects}
    for w in writers.values():
        w.__enter__()

    t0 = time.time()
    try:
        for i, fid in enumerate(frame_ids):
            merged = merge_frame(
                fid, objects, person_keys, readers,
                generated_dir, image_dir, validity_dir, aligner,
            )
            for obj in objects:
                comp = compress_mask_frame(merged[obj], compression_level=6)
                writers[obj].write_frame_compressed(fid, comp)
            if (i + 1) % progress_every == 0:
                dt = time.time() - t0
                rate = (i + 1) / dt
                eta = (len(frame_ids) - i - 1) / max(rate, 1e-6)
                print(f"  [{seq}] {i+1}/{len(frame_ids)} frames "
                      f"({rate:.1f} fps, ETA {eta/60:.1f} min)", flush=True)
    except Exception as e:
        # Cleanup partials
        for w in writers.values():
            try:
                w.__exit__(None, None, None)
            except Exception:
                pass
        for p in new_paths.values():
            if os.path.exists(p):
                os.remove(p)
        for r in readers.values():
            r.close()
        raise

    # Finalize writers
    for w in writers.values():
        w.__exit__(None, None, None)
    for r in readers.values():
        r.close()

    # If output is in-place, .bak originals then atomic mv .new → .shard
    if output_dir is None or os.path.abspath(output_dir) == os.path.abspath(seq_dir):
        for obj in objects:
            orig = join(seq_dir, f"{obj}.shard")
            bak = join(seq_dir, f"{obj}.shard.bak")
            if keep_bak and not os.path.exists(bak):
                shutil.copy2(orig, bak)
            os.replace(new_paths[obj], orig)
    else:
        # Copy meta.json so output_dir is self-contained
        shutil.copy2(join(seq_dir, "meta.json"), join(out_dir, "meta.json"))
        # rename .new → .shard inside output_dir
        for obj in objects:
            os.replace(new_paths[obj], join(out_dir, f"{obj}.shard"))


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--mask_shard_root", default=DEFAULT_SHARD_ROOT)
    ap.add_argument("--validity_root", default=None,
                    help="Override; otherwise auto-find under /scr/juze or /simurgh2")
    ap.add_argument("--generated_root", default=DEFAULT_GENERATED_ROOT,
                    help="Root of fresh YOLO+SAM 720p NPZ outputs.")
    ap.add_argument("--image_root", default=DEFAULT_IMAGE_ROOT)
    ap.add_argument("--reid_model", default=DEFAULT_REID_MODEL)
    ap.add_argument("--output_root", default=None,
                    help="If unset, rewrite in-place atomically.")
    ap.add_argument("--reuse_npz_generated", action="store_true",
                    help="Reuse existing mask_npz_generated outputs; skip inference.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--start_frame", type=int, default=0)
    ap.add_argument("--end_frame", type=int, default=0,
                    help="0 = all frames.")
    ap.add_argument("--max_frames", type=int, default=0,
                    help="0 = no limit. Useful for smoke tests.")
    ap.add_argument("--sim_threshold", type=float, default=0.5)
    ap.add_argument("--no_bak", action="store_true")
    ap.add_argument("--refresh_validity", action="store_true",
                    help="After merge, re-run voxel-consistency check to refresh validity npz.")
    ap.add_argument("--validity_output_root", default="/scr/juze/datasets/HOI-M3/mask_validity_post_merge")
    ap.add_argument("--dry_run", action="store_true",
                    help="Plan only; do not write shards.")
    return ap.parse_args()


def main():
    args = parse_args()

    seq = args.sequence
    seq_shard_dir = join(args.mask_shard_root, seq)
    meta_path = join(seq_shard_dir, "meta.json")
    if not os.path.exists(meta_path):
        print(f"FAIL: no meta.json at {meta_path}", flush=True)
        sys.exit(2)
    with open(meta_path) as f:
        meta = json.load(f)

    if meta["height"] != SHARD_H or meta["width"] != SHARD_W:
        print(f"FAIL: shard not 1080p: {meta['width']}x{meta['height']}", flush=True)
        sys.exit(2)

    objects: List[str] = meta["objects"]
    all_frame_ids: List[int] = meta["frame_ids"]

    start = args.start_frame
    end = args.end_frame if args.end_frame > 0 else (max(all_frame_ids) + 1)
    target_frames = [f for f in all_frame_ids if start <= f < end]
    if args.max_frames > 0:
        target_frames = target_frames[:args.max_frames]

    print(f"[merge_regen] seq={seq} objects={objects} frames={len(target_frames)} "
          f"({target_frames[0]}..{target_frames[-1]})", flush=True)

    validity_dir = resolve_validity_root(seq, args.validity_root)
    generated_dir = join(args.generated_root, seq)
    image_dir = join(args.image_root, seq)

    if not os.path.isdir(generated_dir):
        print(f"FAIL: no generated dir: {generated_dir}", flush=True)
        sys.exit(2)
    if not os.path.isdir(image_dir):
        print(f"FAIL: no image dir: {image_dir}", flush=True)
        sys.exit(2)

    # Sanity: count fresh-NPZ coverage
    gen_files = set()
    for f in os.listdir(generated_dir):
        if f.endswith(".npz"):
            try:
                gen_files.add(int(f[:-4]))
            except ValueError:
                pass
    missing_fresh = [f for f in target_frames if f not in gen_files]
    print(f"  validity_dir={validity_dir}", flush=True)
    print(f"  generated_dir={generated_dir} (have {len(gen_files)} fresh frames, "
          f"{len(missing_fresh)} target frames missing fresh masks)", flush=True)
    if missing_fresh and not args.reuse_npz_generated:
        # In the real pipeline you'd kick off inference for missing frames here.
        # For now we just warn — frames without fresh data simply keep originals.
        print("  WARNING: some frames lack fresh inference; those frames will keep originals.",
              flush=True)

    # ReID aligner
    person_keys = [o for o in objects if o.startswith("person")]
    aligner = None
    if person_keys:
        print(f"  loading ReID model {args.reid_model} ...", flush=True)
        aligner = ReIDAligner(args.reid_model, device=args.device,
                              sim_threshold=args.sim_threshold)

    if args.dry_run:
        print("[dry_run] skipping shard rewrite.", flush=True)
        return

    # Output directory: <output_root>/<seq> when explicit; None → in-place
    output_dir = join(args.output_root, seq) if args.output_root else None

    rewrite_sequence(
        seq=seq,
        seq_dir=seq_shard_dir,
        objects=objects,
        frame_ids=target_frames,
        generated_dir=generated_dir,
        image_dir=image_dir,
        validity_dir=validity_dir,
        aligner=aligner,
        output_dir=output_dir,
        keep_bak=not args.no_bak,
    )
    print(f"[merge_regen] {seq}: shard rewrite done.", flush=True)

    if args.refresh_validity:
        # Delegate to multi_view_mask_check.py
        out_v = join(args.validity_output_root, seq)
        os.makedirs(out_v, exist_ok=True)
        rp = "/simurgh/group/juze/datasets/HOI-M3"
        cmd = [
            sys.executable,
            "/simurgh/u/juze/code/HOIM3_Toolbox/scripts/multi_view_mask_check.py",
            "--root_path", rp,
            "--seq_name", seq,
            "--output_path", args.validity_output_root,
            "--all_views",
            "--mask_format", "shard",
            "--mask_root", args.mask_shard_root,
        ]
        print(f"[merge_regen] refreshing validity: {' '.join(cmd)}", flush=True)
        subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
