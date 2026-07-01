#!/usr/bin/env python3
"""Step 2: Fine-tune YOLO11-seg using WebDataset training shards.

Subclasses SegmentationTrainer to swap the training DataLoader for a
WebDataset-backed IterableDataset.  Validation uses standard YOLO format.

Usage:
    python scripts/yolo_seg/train.py \
        --data /simurgh2/datasets/HOI-M3/yolo_seg_wds/data.yaml \
        --model yolo11m-seg.pt --epochs 50 --batch 32 --device 0,1
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


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="YOLO11-seg training with WebDataset")
    p.add_argument("--data", type=str, default=os.path.join(WDS_ROOT, "data.yaml"))
    p.add_argument("--model", type=str, default="yolo11m-seg.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--lr0", type=float, default=0.001)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--project", type=str, default=os.path.join(WDS_ROOT, "runs"))
    p.add_argument("--name", type=str, default="yolo11m-seg")
    p.add_argument("--resume", type=str, default=None,
                   help="Resume from checkpoint (path to last.pt)")
    p.add_argument("--freeze", type=int, default=None,
                   help="Freeze first N layers of backbone (e.g. 10)")
    p.add_argument("--warmup_epochs", type=float, default=None,
                   help="Override warmup epochs (default: 3.0)")
    return p.parse_args()


# ── Image helpers ─────────────────────────────────────────────────────────────

def _letterbox(img, imgsz, stride=32):
    """Letterbox preserving aspect ratio, pad to stride multiple — matches YOLO val preprocessing.

    For 640×360 input with imgsz=640: produces 384×640 (not 640×640).
    This ensures training and validation see the same spatial layout.
    """
    h, w = img.shape[:2]
    scale = min(imgsz / w, imgsz / h)
    nw, nh = int(w * scale), int(h * scale)
    # Pad to nearest stride multiple (not to square)
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


def _segments_to_mask(segments, mask_h, mask_w, nw, nh, dx, dy, canvas_w, canvas_h):
    """Rasterise normalised polygon segments into a binary mask in letterbox space."""
    mask = np.zeros((mask_h, mask_w), dtype=np.float32)
    for seg in segments:
        pts = np.array(seg, dtype=np.float32)  # (P, 2) normalised in original image
        px = (pts[:, 0] * nw + dx) / canvas_w * mask_w
        py = (pts[:, 1] * nh + dy) / canvas_h * mask_h
        poly = np.column_stack([px, py]).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [poly], 1.0)
    return mask


# ── WebDataset segment dataset ────────────────────────────────────────────────

class WDSSegDataset(IterableDataset):
    """Streams YOLO-seg training samples from WebDataset tar shards."""

    def __init__(self, shard_dir: str, imgsz: int = 640, augment: bool = True, stride: int = 32):
        super().__init__()
        self.urls = sorted(glob.glob(os.path.join(shard_dir, "shard-*.tar")))
        if not self.urls:
            raise FileNotFoundError(f"No shards in {shard_dir}")
        self.imgsz = imgsz
        self.stride = stride
        self.augment = augment
        # Mask dims are computed per-sample based on actual canvas size (not fixed square)

    # ultralytics calls this late in training to disable mosaic augmentation
    def close_mosaic(self, *_a, **_kw):
        pass

    # Stubs for attributes the trainer inspects but that don't apply to streaming data
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

    # ── per-sample processing ─────────────────────────────────────────────

    def _process(self, sample):
        img = cv2.imdecode(np.frombuffer(sample["jpg"], np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        ann = json.loads(sample["json"])
        items = ann.get("annotations", [])
        if not items:
            return None

        # Letterbox (preserves aspect ratio, pads to stride multiple — matches YOLO val)
        img_lb, nw, nh, dx, dy = _letterbox(img, self.imgsz, self.stride)
        canvas_h, canvas_w = img_lb.shape[:2]  # e.g. 384×640 for 16:9
        mask_h, mask_w = canvas_h // 4, canvas_w // 4  # e.g. 96×160

        cls_l, box_l, mask_l = [], [], []
        for a in items:
            cx, cy, w, h = a["bbox"]
            cx_lb = (cx * nw + dx) / canvas_w
            cy_lb = (cy * nh + dy) / canvas_h
            w_lb = w * nw / canvas_w
            h_lb = h * nh / canvas_h

            # Skip degenerate boxes (cause inf loss)
            if w_lb < 1e-4 or h_lb < 1e-4 or w_lb > 1.0 or h_lb > 1.0:
                continue
            if not (0.0 <= cx_lb <= 1.0 and 0.0 <= cy_lb <= 1.0):
                continue

            cls_l.append(a["class_id"])
            box_l.append([cx_lb, cy_lb, w_lb, h_lb])
            mask_l.append(
                _segments_to_mask(a["segments"], mask_h, mask_w,
                                  nw, nh, dx, dy, canvas_w, canvas_h)
            )

        if not cls_l:
            return None

        # ── augmentations (no mosaic) ─────────────────────────────────────
        if self.augment:
            if random.random() < 0.5:  # horizontal flip
                img_lb = np.ascontiguousarray(np.fliplr(img_lb))
                for i in range(len(box_l)):
                    box_l[i][0] = 1.0 - box_l[i][0]
                    mask_l[i] = np.ascontiguousarray(np.fliplr(mask_l[i]))
            img_lb = _hsv_augment(img_lb)

        return {
            "img": torch.from_numpy(img_lb.transpose(2, 0, 1).copy()),  # uint8 CHW
            "cls": torch.tensor(cls_l, dtype=torch.float32).unsqueeze(1),
            "bboxes": torch.tensor(box_l, dtype=torch.float32),
            "masks": torch.from_numpy(np.stack(mask_l)),
        }


# ── Collate + DataLoader wrapper ─────────────────────────────────────────────

def _collate(batch):
    imgs, cls, bboxes, masks, bidx = [], [], [], [], []
    for i, s in enumerate(batch):
        imgs.append(s["img"])
        n = s["cls"].shape[0]
        cls.append(s["cls"])
        bboxes.append(s["bboxes"])
        masks.append(s["masks"])
        bidx.append(torch.full((n,), i, dtype=torch.long))
    mh = batch[0]["masks"].shape[1]
    mw = batch[0]["masks"].shape[2]
    return {
        "img": torch.stack(imgs),
        "cls": torch.cat(cls) if cls else torch.zeros(0, 1),
        "bboxes": torch.cat(bboxes) if bboxes else torch.zeros(0, 4),
        "masks": torch.cat(masks) if masks else torch.zeros(0, mh, mw),
        "batch_idx": torch.cat(bidx) if bidx else torch.zeros(0, dtype=torch.long),
    }


class _WDSLoader(DataLoader):
    """DataLoader with __len__, reset(), and close_mosaic() for ultralytics compat."""
    def __init__(self, dataset, num_batches, **kw):
        super().__init__(dataset, **kw)
        self._nb = num_batches

    def reset(self):
        """Called by trainer at epoch start — no-op for streaming dataset."""
        pass

    def close_mosaic(self, *_a, **_kw):
        """Called by trainer to disable mosaic — no-op."""
        pass

    def __len__(self):
        return self._nb


def _build_wds_loader(shard_dir, batch_size, imgsz=640, workers=4):
    ds = WDSSegDataset(shard_dir, imgsz=imgsz, augment=True)
    nb = len(ds.urls) * SAMPLES_PER_SHARD // max(batch_size, 1)
    return _WDSLoader(
        ds, num_batches=nb,
        batch_size=batch_size, collate_fn=_collate,
        num_workers=workers, pin_memory=True, drop_last=True,
    )


# ── Custom trainer ────────────────────────────────────────────────────────────

class WDSSegTrainer(SegmentationTrainer):
    """SegmentationTrainer that reads training data from WebDataset shards."""

    def get_dataloader(self, dataset_path, batch_size, rank=0, mode="train"):
        if mode == "train":
            return _build_wds_loader(
                dataset_path, batch_size,
                imgsz=self.args.imgsz,
                workers=self.args.workers,
            )
        # Validation uses standard YOLO image/label directories
        return super().get_dataloader(dataset_path, batch_size, rank, mode)

    def plot_training_labels(self):
        """Skip — not available for streaming dataset."""
        pass

    def plot_training_samples(self, batch, ni):
        """Skip — streaming dataset has no im_file paths."""
        pass

    def optimizer_step(self):
        """Clip gradients before optimizer step to prevent inf/nan loss."""
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        super().optimizer_step()

    def preprocess_batch(self, batch):
        """Sanitize batch: clamp bbox values and skip if any NaN/inf."""
        batch = super().preprocess_batch(batch)
        # Clamp bboxes to valid range
        if "bboxes" in batch and batch["bboxes"].numel() > 0:
            batch["bboxes"] = batch["bboxes"].clamp(0.0, 1.0)
            # Remove any NaN/inf entries
            valid = torch.isfinite(batch["bboxes"]).all(dim=1)
            if not valid.all():
                batch["bboxes"] = batch["bboxes"][valid]
                batch["cls"] = batch["cls"][valid]
                batch["masks"] = batch["masks"][valid]
                batch["batch_idx"] = batch["batch_idx"][valid]
        return batch


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Enable wandb logging (ultralytics has built-in wandb callback)
    os.environ.setdefault("WANDB_PROJECT", "hoim3-yolo-seg")

    overrides = {
        "task": "segment",
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
        "overlap_mask": False,  # our masks are per-instance (N,H,W), not overlap-indexed (B,H,W)
    }
    if args.resume:
        overrides["resume"] = args.resume
    if args.freeze is not None:
        overrides["freeze"] = args.freeze
    if args.warmup_epochs is not None:
        overrides["warmup_epochs"] = args.warmup_epochs

    trainer = WDSSegTrainer(overrides=overrides)
    trainer.train()


if __name__ == "__main__":
    main()
