#!/usr/bin/env python3
"""Train a separate YOLO11-seg model for each object class (binary: object vs background).

Person is excluded (already works well). Each model only needs to distinguish
one object class, making training much easier and more reliable.

Usage:
    # Train one object
    python scripts/yolo_seg/train_per_object.py --target_class bed --epochs 30

    # Train all objects
    python scripts/yolo_seg/train_per_object.py --all --epochs 30

    # Train specific objects
    python scripts/yolo_seg/train_per_object.py --target_class bed television yogamat --epochs 30
"""

import argparse
import glob
import json
import os
import random
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

import webdataset as wds  # noqa: E402
from ultralytics.models.yolo.segment import SegmentationTrainer  # noqa: E402

from scripts.yolo_seg.config import WDS_ROOT, SAMPLES_PER_SHARD  # noqa: E402
from scripts.yolo_seg.train import (  # noqa: E402
    _letterbox, _hsv_augment, _segments_to_mask, _collate, _WDSLoader,
)


def parse_args():
    p = argparse.ArgumentParser(description="Per-object YOLO11-seg training")
    p.add_argument("--target_class", nargs="+", default=None,
                   help="Object class name(s) to train (e.g. bed television)")
    p.add_argument("--all", action="store_true",
                   help="Train all 87 object classes sequentially")
    p.add_argument("--shard_dir", type=str, default=os.path.join(WDS_ROOT, "train_shards"))
    p.add_argument("--val_dir", type=str, default=os.path.join(WDS_ROOT, "val"))
    p.add_argument("--model", type=str, default="yolo11n-seg.pt",
                   help="Base model (nano is enough for binary)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--freeze", type=int, default=10)
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--project", type=str, default=os.path.join(WDS_ROOT, "runs_per_object"))
    return p.parse_args()


# ── Per-object dataset: filters to single class ──────────────────────────────

class PerObjectDataset(IterableDataset):
    """WebDataset filtered to a single target object class (binary: class 0)."""

    def __init__(self, shard_dir, target_class, class_mapping, imgsz=640, augment=True, stride=32):
        super().__init__()
        self.urls = sorted(glob.glob(os.path.join(shard_dir, "shard-*.tar")))
        if not self.urls:
            raise FileNotFoundError(f"No shards in {shard_dir}")
        self.target_class = target_class
        self.target_id = class_mapping.get(target_class, -1)
        self.imgsz = imgsz
        self.stride = stride
        self.augment = augment

    def close_mosaic(self, *_a, **_kw):
        pass

    @property
    def labels(self):
        return []

    def __len__(self):
        # Rough estimate: ~10% of samples contain any given object
        return max(len(self.urls) * SAMPLES_PER_SHARD // 10, 1000)

    def __iter__(self):
        pipeline = wds.WebDataset(self.urls, shardshuffle=self.augment)
        if self.augment:
            pipeline = pipeline.shuffle(5000)
        for sample in pipeline:
            out = self._process(sample)
            if out is not None:
                yield out

    def _process(self, sample):
        img = cv2.imdecode(np.frombuffer(sample["jpg"], np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        ann = json.loads(sample["json"])
        items = ann.get("annotations", [])

        # Filter to target class only
        target_items = [a for a in items if a["class_id"] == self.target_id]
        if not target_items:
            return None  # Skip samples without this object

        img_lb, nw, nh, dx, dy = _letterbox(img, self.imgsz, self.stride)
        canvas_h, canvas_w = img_lb.shape[:2]
        mask_h, mask_w = canvas_h // 4, canvas_w // 4

        cls_l, box_l, mask_l = [], [], []
        for a in target_items:
            cx, cy, w, h = a["bbox"]
            cx_lb = (cx * nw + dx) / canvas_w
            cy_lb = (cy * nh + dy) / canvas_h
            w_lb = w * nw / canvas_w
            h_lb = h * nh / canvas_h

            if w_lb < 1e-4 or h_lb < 1e-4 or w_lb > 1.0 or h_lb > 1.0:
                continue
            if not (0.0 <= cx_lb <= 1.0 and 0.0 <= cy_lb <= 1.0):
                continue

            cls_l.append(0)  # Binary: always class 0
            box_l.append([cx_lb, cy_lb, w_lb, h_lb])
            mask_l.append(
                _segments_to_mask(a["segments"], mask_h, mask_w,
                                  nw, nh, dx, dy, canvas_w, canvas_h)
            )

        if not cls_l:
            return None

        if self.augment:
            if random.random() < 0.5:
                img_lb = np.ascontiguousarray(np.fliplr(img_lb))
                for i in range(len(box_l)):
                    box_l[i][0] = 1.0 - box_l[i][0]
                    mask_l[i] = np.ascontiguousarray(np.fliplr(mask_l[i]))
            img_lb = _hsv_augment(img_lb)

        return {
            "img": torch.from_numpy(img_lb.transpose(2, 0, 1).copy()),
            "cls": torch.tensor(cls_l, dtype=torch.float32).unsqueeze(1),
            "bboxes": torch.tensor(box_l, dtype=torch.float32),
            "masks": torch.from_numpy(np.stack(mask_l)),
        }


def _build_per_object_loader(shard_dir, target_class, class_mapping, batch_size, imgsz=640, workers=2):
    ds = PerObjectDataset(shard_dir, target_class, class_mapping, imgsz=imgsz, augment=True)
    nb = len(ds) // max(batch_size, 1)
    return _WDSLoader(
        ds, num_batches=nb,
        batch_size=batch_size, collate_fn=_collate,
        num_workers=workers, pin_memory=True, drop_last=True,
    )


# ── Per-object val data (YOLO format) ────────────────────────────────────────

def create_per_object_val(val_dir, target_class_name, target_class_id, output_dir):
    """Create a binary val set by filtering labels to target class only."""
    src_img = os.path.join(val_dir, "images")
    src_lbl = os.path.join(val_dir, "labels")
    dst_img = os.path.join(output_dir, "images")
    dst_lbl = os.path.join(output_dir, "labels")
    os.makedirs(dst_img, exist_ok=True)
    os.makedirs(dst_lbl, exist_ok=True)

    count = 0
    for img_file in os.listdir(src_img):
        lbl_file = img_file.replace(".jpg", ".txt")
        lbl_path = os.path.join(src_lbl, lbl_file)
        if not os.path.exists(lbl_path):
            continue

        # Filter label lines to target class
        with open(lbl_path) as f:
            lines = f.readlines()

        filtered = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            cls_id = int(parts[0])
            if cls_id == target_class_id:
                # Remap to class 0
                filtered.append("0 " + " ".join(parts[1:]) + "\n")

        if filtered:
            # Symlink image, write filtered label
            src = os.path.join(src_img, img_file)
            dst = os.path.join(dst_img, img_file)
            if not os.path.exists(dst):
                os.symlink(src, dst)
            with open(os.path.join(dst_lbl, lbl_file), "w") as f:
                f.writelines(filtered)
            count += 1

    return count


# ── Trainer ───────────────────────────────────────────────────────────────────

class PerObjectTrainer(SegmentationTrainer):
    _target_class = None
    _class_mapping = None
    _shard_dir = None

    def get_dataloader(self, dataset_path, batch_size, rank=0, mode="train"):
        if mode == "train":
            return _build_per_object_loader(
                self._shard_dir, self._target_class, self._class_mapping,
                batch_size, imgsz=self.args.imgsz, workers=self.args.workers,
            )
        return super().get_dataloader(dataset_path, batch_size, rank, mode)

    def plot_training_labels(self):
        pass

    def plot_training_samples(self, batch, ni):
        pass

    def optimizer_step(self):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        super().optimizer_step()

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        if "bboxes" in batch and batch["bboxes"].numel() > 0:
            batch["bboxes"] = batch["bboxes"].clamp(0.0, 1.0)
        return batch


# ── Main ──────────────────────────────────────────────────────────────────────

def train_single_object(target_class, args, class_mapping):
    """Train one model for one object class."""
    print(f"\n{'='*60}")
    print(f"Training: {target_class}")
    print(f"{'='*60}")

    target_id = class_mapping.get(target_class, -1)
    if target_id == -1:
        print(f"  SKIP: {target_class} not in class mapping")
        return

    # Create per-object val set
    val_obj_dir = os.path.join(args.project, f"val_{target_class}")
    n_val = create_per_object_val(args.val_dir, target_class, target_id, val_obj_dir)
    print(f"  Val images with {target_class}: {n_val}")
    if n_val < 5:
        print(f"  SKIP: too few val images")
        return

    # Write binary data.yaml
    yaml_path = os.path.join(args.project, f"data_{target_class}.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {args.project}\n")
        f.write(f"train: {args.shard_dir}\n")
        f.write(f"val: val_{target_class}/images\n")
        f.write("nc: 1\n")
        f.write("names:\n")
        f.write(f"  0: {target_class}\n")

    # Set class info on trainer
    PerObjectTrainer._target_class = target_class
    PerObjectTrainer._class_mapping = class_mapping
    PerObjectTrainer._shard_dir = args.shard_dir

    overrides = {
        "task": "segment",
        "model": args.model,
        "data": yaml_path,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "lr0": args.lr0,
        "workers": args.workers,
        "project": args.project,
        "name": target_class,
        "overlap_mask": False,
        "freeze": args.freeze,
        "exist_ok": True,
    }

    trainer = PerObjectTrainer(overrides=overrides)
    trainer.train()
    print(f"  Done: {target_class}")


def main():
    args = parse_args()
    os.makedirs(args.project, exist_ok=True)

    # Load class mapping
    mapping_path = os.path.join(WDS_ROOT, "class_mapping.json")
    with open(mapping_path) as f:
        cm = json.load(f)
    class_to_id = cm["class_to_id"]
    class_names = cm["class_names"]

    # Determine target classes
    if args.all:
        # All objects except person (class 0)
        targets = [n for n in class_names if n != "person"]
    elif args.target_class:
        targets = args.target_class
    else:
        print("Specify --target_class or --all")
        return

    print(f"Training {len(targets)} object models")
    print(f"Base model: {args.model}")
    print(f"Freeze: {args.freeze} layers")
    print(f"Epochs: {args.epochs}")

    for target in targets:
        train_single_object(target, args, class_to_id)

    print(f"\n{'='*60}")
    print(f"All done! Models saved in {args.project}")


if __name__ == "__main__":
    main()
