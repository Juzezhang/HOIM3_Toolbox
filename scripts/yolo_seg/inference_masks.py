#!/usr/bin/env python3
"""Generate mask_npz for test sequences using YOLO seg + ReID pipeline.

Optimized: GPU batch mask resize, fast ReID matching, pipelined I/O.

Usage:
    python scripts/yolo_seg/inference_masks.py --gpu 0 --split 0
    python scripts/yolo_seg/inference_masks.py --gpu 1 --split 1
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, Future

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

from ultralytics import YOLO
from scripts.yolo_seg.train_reid import ReIDModel
from scripts.yolo_seg.config import load_sequence_contents, get_test_sequences

IMAGE_ROOT = "/simurgh2/datasets/HOI-M3/images"
OUTPUT_ROOT = "/simurgh2/datasets/HOI-M3/mask_npz_generated"
SEG_MODEL_PATH = "/simurgh2/datasets/HOI-M3/yolo_std/runs/native5/weights/best.pt"
REID_MODEL_PATH = "/simurgh2/datasets/HOI-M3/reid_data/runs/best_reid.pt"
CLASS_MAPPING_PATH = "/simurgh2/datasets/HOI-M3/yolo_std/class_mapping.json"

MASK_H, MASK_W = 720, 1280
NUM_VIEWS = 42


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--split", type=int, default=0, choices=[0, 1])
    p.add_argument("--sequences", nargs="+", default=None)
    p.add_argument("--start_frame", type=int, default=0)
    p.add_argument("--end_frame", type=int, default=0)
    return p.parse_args()


def load_models(device):
    print("Loading models...", flush=True)
    seg_model = YOLO(SEG_MODEL_PATH)

    reid_ckpt = torch.load(REID_MODEL_PATH, map_location=device, weights_only=False)
    reid_model = ReIDModel(num_classes=441, embed_dim=256).to(device)
    reid_model.load_state_dict(reid_ckpt["model_state_dict"])
    reid_model.eval()

    reid_transform = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    with open(CLASS_MAPPING_PATH) as f:
        cm = json.load(f)
    id_to_name = {i: name for i, name in enumerate(cm["class_names"])}

    return seg_model, reid_model, reid_transform, id_to_name


def load_views_parallel(seq_img_dir, frame_idx, pool):
    def _read(v):
        path = os.path.join(seq_img_dir, str(v), f"{frame_idx:06d}.jpg")
        return v, cv2.imread(path)
    futures = [pool.submit(_read, v) for v in range(NUM_VIEWS)]
    views = {}
    for f in futures:
        v, img = f.result()
        if img is not None:
            views[v] = img
    return views


def process_frame(views_frames, seq_info, seg_model, reid_model,
                  reid_transform, id_to_name, device):
    """Optimized: minimal CPU work, batch GPU ops."""
    num_humans = seq_info["num_humans"]
    expected_objects = set(seq_info["objects"])

    sorted_views = sorted(views_frames.keys())
    frames_list = [views_frames[v] for v in sorted_views]

    # YOLO batch predict
    results = seg_model.predict(frames_list, imgsz=640, conf=0.25, verbose=False)

    # Parse results — only keep what we need, defer mask resize
    # For each view: best object det per class, all person dets
    per_view_objects = {}   # view -> {cls_name: (conf, mask_raw)}
    person_entries = []     # (view_idx, conf, mask_raw, crop_rgb)

    for vi, (view_idx, r) in enumerate(zip(sorted_views, results)):
        if r.boxes is None or len(r.boxes) == 0:
            continue

        frame = views_frames[view_idx]
        img_h, img_w = frame.shape[:2]
        boxes = r.boxes
        n_det = len(boxes)

        # Batch extract: cls, conf, xyxy
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        xyxys = boxes.xyxy.cpu().numpy().astype(int)

        # Get all masks at once (they're already on CPU after .data)
        masks_raw = r.masks.data.cpu().numpy() if r.masks is not None else None

        obj_best = {}
        for j in range(n_det):
            cls_name = id_to_name.get(cls_ids[j], f"class{cls_ids[j]}")
            conf = confs[j]
            mask = masks_raw[j] if masks_raw is not None and j < len(masks_raw) else None

            if cls_name == "person":
                x1, y1, x2, y2 = xyxys[j]
                crop = frame[max(0, y1):min(img_h, y2), max(0, x1):min(img_w, x2)]
                if crop.size > 0:
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    person_entries.append((view_idx, conf, mask, crop_rgb))
            elif cls_name in expected_objects:
                if cls_name not in obj_best or conf > obj_best[cls_name][0]:
                    obj_best[cls_name] = (conf, mask)

        per_view_objects[view_idx] = obj_best

    # ReID: only keep top-1 person per view, then match across views
    # First, select up to 1 best person per view (reduces from 84 to 42)
    best_person_per_view = {}  # view -> (conf, mask, crop_rgb)
    # Actually keep top num_humans per view for better matching
    from collections import defaultdict
    view_persons = defaultdict(list)
    for (v, conf, mask, crop) in person_entries:
        view_persons[v].append((conf, mask, crop))

    # Keep top min(num_humans, detected) per view
    filtered_persons = []
    for v in sorted(view_persons.keys()):
        persons = sorted(view_persons[v], key=lambda x: -x[0])[:num_humans]
        for conf, mask, crop in persons:
            filtered_persons.append((v, conf, mask, crop))

    # ReID matching
    person_assignments = {}  # idx_in_filtered -> person_id
    if filtered_persons and num_humans > 0:
        # Fast batch transform: cv2 resize + GPU normalize (170x faster than PIL)
        n_crops = len(filtered_persons)
        crops_arr = np.zeros((n_crops, 3, 256, 128), dtype=np.float32)
        for ci, (_, _, _, crop_rgb) in enumerate(filtered_persons):
            resized = cv2.resize(crop_rgb, (128, 256)).transpose(2, 0, 1).astype(np.float32) / 255.0
            crops_arr[ci] = resized
        crops_batch = torch.from_numpy(crops_arr).to(device)
        _mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        _std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        crops_batch = (crops_batch - _mean) / _std

        with torch.no_grad():
            embs, _ = reid_model(crops_batch)
        embs = embs.cpu().numpy()

        n = len(embs)
        if n <= num_humans:
            labels = list(range(n))
        else:
            # Fast greedy assignment using cosine similarity
            sim = embs @ embs.T
            n_clusters = min(num_humans, n)
            # Use scipy for fast hierarchical clustering
            from scipy.cluster.hierarchy import fcluster, linkage
            from scipy.spatial.distance import squareform
            dist = 1.0 - sim
            np.fill_diagonal(dist, 0)
            # Make symmetric and convert to condensed form
            dist = (dist + dist.T) / 2
            condensed = squareform(dist, checks=False)
            Z = linkage(condensed, method='average')
            labels = fcluster(Z, t=n_clusters, criterion='maxclust') - 1  # 0-indexed

        for idx in range(n):
            person_assignments[idx] = int(labels[idx])

    # Build output masks — batch resize on GPU
    masks_out = {}
    for obj_name in expected_objects:
        masks_out[obj_name] = np.zeros((NUM_VIEWS, MASK_H, MASK_W), dtype=np.uint8)
    for i in range(num_humans):
        masks_out[f"person{i}"] = np.zeros((NUM_VIEWS, MASK_H, MASK_W), dtype=np.uint8)

    # Collect all masks that need resize, batch them
    resize_tasks = []  # (dest_key, view_idx, mask_raw)

    for view_idx, obj_best in per_view_objects.items():
        for cls_name, (conf, mask) in obj_best.items():
            if mask is not None:
                resize_tasks.append((cls_name, view_idx, mask))

    for idx, (v, conf, mask, crop) in enumerate(filtered_persons):
        pid = person_assignments.get(idx)
        if pid is not None and mask is not None:
            pname = f"person{pid}"
            if pname in masks_out:
                resize_tasks.append((pname, v, mask))

    # Batch resize all masks on GPU
    if resize_tasks:
        raw_masks = [t[2] for t in resize_tasks]
        # Stack and resize on GPU
        batch = torch.from_numpy(np.stack(raw_masks)).unsqueeze(1).float().to(device)
        resized = F.interpolate(batch, size=(MASK_H, MASK_W), mode='bilinear',
                                align_corners=False)
        resized = (resized.squeeze(1) > 0.5).byte().cpu().numpy() * 255

        for i, (key, view_idx, _) in enumerate(resize_tasks):
            masks_out[key][view_idx] = resized[i]

    return masks_out


def process_sequence(seq_name, seq_info, seg_model, reid_model, reid_transform,
                     id_to_name, device, args):
    seq_img_dir = os.path.join(IMAGE_ROOT, seq_name)
    out_dir = os.path.join(OUTPUT_ROOT, seq_name)
    os.makedirs(out_dir, exist_ok=True)

    view0_dir = os.path.join(seq_img_dir, "0")
    all_frames = sorted([int(f.replace(".jpg", "")) for f in os.listdir(view0_dir)
                         if f.endswith(".jpg")])
    if not all_frames:
        return

    start = args.start_frame
    end = args.end_frame if args.end_frame > 0 else max(all_frames) + 1
    target_frames = [f for f in all_frames if start <= f < end]

    done = set()
    for f in os.listdir(out_dir):
        if f.endswith(".npz"):
            done.add(int(f.replace(".npz", "")))
    todo = [f for f in target_frames if f not in done]

    if not todo:
        print(f"  {seq_name}: done ({len(target_frames)} frames)", flush=True)
        return

    print(f"  {seq_name}: {len(todo)}/{len(target_frames)} frames, "
          f"{seq_info['num_humans']}H {len(seq_info['objects'])}O", flush=True)

    io_pool = ThreadPoolExecutor(max_workers=16)
    # 4 save threads: each takes ~1s, so 4 threads give ~0.25s effective/frame
    save_pool = ThreadPoolExecutor(max_workers=4)
    pending_saves = []

    # Pipeline: prefetch next frame while processing current
    prefetch = io_pool.submit(load_views_parallel, seq_img_dir, todo[0], io_pool)

    for i, frame_idx in enumerate(tqdm(todo, desc=f"  {seq_name}", leave=True, miniters=50)):
        views_frames = prefetch.result()

        # Prefetch next
        if i + 1 < len(todo):
            prefetch = io_pool.submit(load_views_parallel, seq_img_dir, todo[i + 1], io_pool)

        # Clean up completed saves (don't block, just check for errors)
        still_pending = []
        for fut in pending_saves:
            if fut.done():
                fut.result()  # raise if error
            else:
                still_pending.append(fut)
        pending_saves = still_pending

        # Backpressure: if too many pending saves, wait for oldest
        while len(pending_saves) >= 8:
            pending_saves[0].result()
            pending_saves.pop(0)

        if not views_frames:
            continue

        masks = process_frame(
            views_frames, seq_info, seg_model, reid_model,
            reid_transform, id_to_name, device
        )

        out_path = os.path.join(out_dir, f"{frame_idx:06d}.npz")
        pending_saves.append(save_pool.submit(np.savez_compressed, out_path, **masks))

    # Wait for all saves
    for fut in pending_saves:
        fut.result()
    save_pool.shutdown(wait=True)
    io_pool.shutdown(wait=False)
    print(f"  {seq_name}: done!", flush=True)


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"GPU: {args.gpu}", flush=True)

    seg_model, reid_model, reid_transform, id_to_name = load_models(device)

    sc = load_sequence_contents()
    test_seqs = get_test_sequences(sc)

    if args.sequences:
        # Allow specifying sequences directly, even if they have existing shards
        requested = set(args.sequences)
        test_seqs = [s for s in test_seqs if s in requested]
        extra = sorted(s for s in requested if s in sc and s not in test_seqs)
        test_seqs = sorted(set(test_seqs + extra))
    else:
        mid = len(test_seqs) // 2
        if args.split == 0:
            test_seqs = test_seqs[:mid + (len(test_seqs) % 2)]
        else:
            test_seqs = test_seqs[mid + (len(test_seqs) % 2):]

    print(f"Sequences ({len(test_seqs)}): {test_seqs}", flush=True)

    for seq_name in test_seqs:
        process_sequence(seq_name, sc[seq_name], seg_model, reid_model,
                         reid_transform, id_to_name, device, args)

    print("All done!", flush=True)


if __name__ == "__main__":
    main()
