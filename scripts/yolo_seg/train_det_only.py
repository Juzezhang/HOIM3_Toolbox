#!/usr/bin/env python3
"""Detection-only YOLO11 training (no segmentation masks) for comparison.

Uses the same WebDataset but only trains bbox + class, ignoring masks.
This helps diagnose whether the seg head is causing training instability.

Usage:
    python scripts/yolo_seg/train_det_only.py \
        --data /simurgh2/datasets/HOI-M3/yolo_seg_wds/data.yaml \
        --model yolo11m.pt --epochs 100 --batch 16 --device 0
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
from ultralytics.models.yolo.detect import DetectionTrainer  # noqa: E402

from scripts.yolo_seg.config import WDS_ROOT, SAMPLES_PER_SHARD  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Detection-only YOLO11 training")
    p.add_argument("--data", type=str, default=os.path.join(WDS_ROOT, "data.yaml"))
    p.add_argument("--model", type=str, default="yolo11m.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--project", type=str, default=os.path.join(WDS_ROOT, "runs"))
    p.add_argument("--name", type=str, default="det_only")
    return p.parse_args()


# ── Letterbox (same as train.py — matches YOLO val) ──────────────────────────

def _letterbox(img, imgsz, stride=32):
    h, w = img.shape[:2]
    scale = min(imgsz / w, imgsz / h)
    nw, nh = int(w * scale), int(h * scale)
    canvas_w = int(np.ceil(nw / stride) * stride)
    canvas_h = int(np.ceil(nh / stride) * stride)
    if (nw, nh) != (w, h):
        img = cv2.resize(img, (nw, nh))
    canvas = np.full((canvas_h, canvas_w, 3), 114, dtype=np.uint8)
    dx = (canvas_w - nw) // 2
    dy = (canvas_h - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = img
    return canvas, nw, nh, dx, dy


def _hsv_augment(img, hgain=0.015, sgain=0.7, vgain=0.4):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    x = np.arange(256, dtype=np.float32)
    lut_h = ((x * r[0]) % 180).astype(np.uint8)
    lut_s = np.clip(x * r[1], 0, 255).astype(np.uint8)
    lut_v = np.clip(x * r[2], 0, 255).astype(np.uint8)
    hsv = cv2.merge((cv2.LUT(hue, lut_h), cv2.LUT(sat, lut_s), cv2.LUT(val, lut_v)))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# ── Detection-only dataset (no masks) ─────────────────────────────────────────

class WDSDetDataset(IterableDataset):
    """WebDataset for detection-only training (bbox + class, no masks)."""

    def __init__(self, shard_dir, imgsz=640, augment=True, stride=32):
        super().__init__()
        self.urls = sorted(glob.glob(os.path.join(shard_dir, "shard-*.tar")))
        if not self.urls:
            raise FileNotFoundError(f"No shards in {shard_dir}")
        self.imgsz = imgsz
        self.stride = stride
        self.augment = augment

    def close_mosaic(self, *_a, **_kw):
        pass

    @property
    def labels(self):
        return []

    def __len__(self):
        return len(self.urls) * SAMPLES_PER_SHARD

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
        if not items:
            return None

        img_lb, nw, nh, dx, dy = _letterbox(img, self.imgsz, self.stride)
        canvas_h, canvas_w = img_lb.shape[:2]

        cls_l, box_l = [], []
        for a in items:
            cx, cy, w, h = a["bbox"]
            cx_lb = (cx * nw + dx) / canvas_w
            cy_lb = (cy * nh + dy) / canvas_h
            w_lb = w * nw / canvas_w
            h_lb = h * nh / canvas_h

            if w_lb < 1e-4 or h_lb < 1e-4 or w_lb > 1.0 or h_lb > 1.0:
                continue
            if not (0.0 <= cx_lb <= 1.0 and 0.0 <= cy_lb <= 1.0):
                continue

            cls_l.append(a["class_id"])
            box_l.append([cx_lb, cy_lb, w_lb, h_lb])

        if not cls_l:
            return None

        if self.augment:
            if random.random() < 0.5:
                img_lb = np.ascontiguousarray(np.fliplr(img_lb))
                for i in range(len(box_l)):
                    box_l[i][0] = 1.0 - box_l[i][0]
            img_lb = _hsv_augment(img_lb)

        return {
            "img": torch.from_numpy(img_lb.transpose(2, 0, 1).copy()),
            "cls": torch.tensor(cls_l, dtype=torch.float32).unsqueeze(1),
            "bboxes": torch.tensor(box_l, dtype=torch.float32),
        }


# ── Collate + DataLoader ─────────────────────────────────────────────────────

def _collate(batch):
    imgs, cls, bboxes, bidx = [], [], [], []
    for i, s in enumerate(batch):
        imgs.append(s["img"])
        n = s["cls"].shape[0]
        cls.append(s["cls"])
        bboxes.append(s["bboxes"])
        bidx.append(torch.full((n,), i, dtype=torch.long))
    return {
        "img": torch.stack(imgs),
        "cls": torch.cat(cls) if cls else torch.zeros(0, 1),
        "bboxes": torch.cat(bboxes) if bboxes else torch.zeros(0, 4),
        "batch_idx": torch.cat(bidx) if bidx else torch.zeros(0, dtype=torch.long),
    }


class _WDSLoader(DataLoader):
    def __init__(self, dataset, num_batches, **kw):
        super().__init__(dataset, **kw)
        self._nb = num_batches

    def reset(self):
        pass

    def close_mosaic(self, *_a, **_kw):
        pass

    def __len__(self):
        return self._nb


def _build_wds_loader(shard_dir, batch_size, imgsz=640, workers=2):
    ds = WDSDetDataset(shard_dir, imgsz=imgsz, augment=True)
    nb = len(ds.urls) * SAMPLES_PER_SHARD // max(batch_size, 1)
    return _WDSLoader(
        ds, num_batches=nb,
        batch_size=batch_size, collate_fn=_collate,
        num_workers=workers, pin_memory=True, drop_last=True,
    )


# ── Custom trainer ────────────────────────────────────────────────────────────

class WDSDetTrainer(DetectionTrainer):
    """DetectionTrainer with WebDataset-backed training DataLoader."""

    def get_dataloader(self, dataset_path, batch_size, rank=0, mode="train"):
        if mode == "train":
            return _build_wds_loader(
                dataset_path, batch_size,
                imgsz=self.args.imgsz,
                workers=self.args.workers,
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
            valid = torch.isfinite(batch["bboxes"]).all(dim=1)
            if not valid.all():
                batch["bboxes"] = batch["bboxes"][valid]
                batch["cls"] = batch["cls"][valid]
                batch["batch_idx"] = batch["batch_idx"][valid]
        return batch


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    overrides = {
        "task": "detect",
        "model": args.model,
        "data": args.data,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "lr0": args.lr0,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
    }

    trainer = WDSDetTrainer(overrides=overrides)
    trainer.train()


if __name__ == "__main__":
    main()
