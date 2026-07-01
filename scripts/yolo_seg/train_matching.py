#!/usr/bin/env python3
"""Step 5: Train cross-view person ID matching network.

Architecture:
  ResNet50 backbone  →  RoIAlign per detection  →  Transformer encoder
  →  pairwise affinity matrix  →  BCE loss

At inference, the affinity matrix is thresholded and connected components
give consistent person IDs across camera views.

Usage:
    python scripts/yolo_seg/train_matching.py --epochs 30 --batch 32 --device 0
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
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.ops import RoIAlign
from tqdm import tqdm

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

import webdataset as wds  # noqa: E402

from scripts.yolo_seg.config import WDS_ROOT, NUM_VIEWS, SAMPLES_PER_SHARD  # noqa: E402


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train cross-view person matching network")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--feat_dim", type=int, default=256)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--roi_size", type=int, default=7)
    p.add_argument("--val_split", type=float, default=0.1,
                   help="Fraction of shards for validation")
    p.add_argument("--save_dir", type=str,
                   default=os.path.join(WDS_ROOT, "matching_runs"))
    return p.parse_args()


# ── Dataset ───────────────────────────────────────────────────────────────────

class MatchingDataset(IterableDataset):
    """Streams matching samples from WebDataset shards produced by prepare_matching_data."""

    def __init__(self, shard_urls: list[str], augment: bool = True):
        super().__init__()
        self.urls = shard_urls
        self.augment = augment

    def __iter__(self):
        pipeline = wds.WebDataset(self.urls, shardshuffle=self.augment)
        if self.augment:
            pipeline = pipeline.shuffle(2000)
        for sample in pipeline:
            out = self._process(sample)
            if out is not None:
                yield out

    def _process(self, sample):
        img = cv2.imdecode(np.frombuffer(sample["jpg"], np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        ann = json.loads(sample["json"])
        persons = ann.get("persons", [])
        if not persons:
            return None

        h, w = img.shape[:2]

        boxes, view_ids, person_ids = [], [], []
        for person in persons:
            pid = person["person_id"]
            for det in person["detections"]:
                cx, cy, bw, bh = det["bbox"]
                x1 = max((cx - bw / 2) * w, 0)
                y1 = max((cy - bh / 2) * h, 0)
                x2 = min((cx + bw / 2) * w, w)
                y2 = min((cy + bh / 2) * h, h)
                boxes.append([x1, y1, x2, y2])
                view_ids.append(det["view_idx"])
                person_ids.append(pid)

        if len(boxes) < 2:
            return None

        N = len(boxes)
        gt = torch.zeros(N, N, dtype=torch.float32)
        for i in range(N):
            for j in range(N):
                if person_ids[i] == person_ids[j]:
                    gt[i, j] = 1.0

        return {
            "img": torch.from_numpy(img.transpose(2, 0, 1).copy()).float() / 255.0,
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "view_ids": torch.tensor(view_ids, dtype=torch.long),
            "person_ids": torch.tensor(person_ids, dtype=torch.long),
            "gt_affinity": gt,
        }


def _matching_collate(batch):
    """Return list of samples — model handles per-sample due to variable N."""
    return batch


# ── Model ─────────────────────────────────────────────────────────────────────

class PersonMatchingNetwork(nn.Module):
    """ResNet50 + RoIAlign + Transformer → pairwise affinity."""

    def __init__(self, feat_dim=256, num_heads=8, num_layers=4,
                 roi_size=7, num_views=NUM_VIEWS):
        super().__init__()

        # Backbone — freeze first two stages
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        for name, param in backbone.named_parameters():
            if "layer3" not in name and "layer4" not in name:
                param.requires_grad = False

        self.backbone = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
        )

        self.roi_align = RoIAlign(
            output_size=(roi_size, roi_size),
            spatial_scale=1.0 / 32,
            sampling_ratio=-1,
        )

        self.feat_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048 * roi_size * roi_size, feat_dim),
            nn.ReLU(inplace=True),
        )

        self.view_embed = nn.Embedding(num_views, feat_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim, nhead=num_heads,
            dim_feedforward=feat_dim * 4, dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.affinity_proj = nn.Linear(feat_dim, feat_dim)

    def forward(self, img, boxes, view_ids):
        """
        Args:
            img: (3, H, W) single composite image tensor
            boxes: (N, 4) [x1, y1, x2, y2] pixel coords
            view_ids: (N,) camera-view indices
        Returns:
            affinity: (N, N) sigmoid similarity matrix
        """
        feat_map = self.backbone(img.unsqueeze(0))  # (1, 2048, H/32, W/32)

        # RoIAlign expects (K, 5) with [batch_idx, x1, y1, x2, y2]
        roi_in = torch.cat([
            torch.zeros(boxes.shape[0], 1, device=boxes.device),
            boxes,
        ], dim=1)
        roi_feats = self.roi_align(feat_map, roi_in)  # (N, 2048, roi, roi)

        feats = self.feat_proj(roi_feats)  # (N, feat_dim)
        feats = feats + self.view_embed(view_ids)

        feats = self.transformer(feats.unsqueeze(0)).squeeze(0)  # (N, feat_dim)

        proj = self.affinity_proj(feats)
        affinity = torch.sigmoid(proj @ proj.T)
        return affinity


# ── Training / evaluation ────────────────────────────────────────────────────

def _train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, n = 0.0, 0
    for batch_list in tqdm(loader, desc="  train", leave=False):
        for sample in batch_list:
            aff = model(
                sample["img"].to(device),
                sample["boxes"].to(device),
                sample["view_ids"].to(device),
            )
            loss = nn.functional.binary_cross_entropy(
                aff, sample["gt_affinity"].to(device),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def _eval(model, loader, device):
    model.eval()
    total_loss, correct, total, n = 0.0, 0, 0, 0
    for batch_list in loader:
        for sample in batch_list:
            aff = model(
                sample["img"].to(device),
                sample["boxes"].to(device),
                sample["view_ids"].to(device),
            )
            gt = sample["gt_affinity"].to(device)
            total_loss += nn.functional.binary_cross_entropy(aff, gt).item()
            correct += ((aff > 0.5).float() == gt).sum().item()
            total += gt.numel()
            n += 1
    return total_loss / max(n, 1), correct / max(total, 1)


# ── Inference utility ─────────────────────────────────────────────────────────

def assign_ids(affinity: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Connected components on thresholded affinity → consistent person IDs."""
    N = affinity.shape[0]
    parent = list(range(N))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(N):
        for j in range(i + 1, N):
            if affinity[i, j] > threshold:
                union(i, j)

    root_map: dict[int, int] = {}
    labels = np.zeros(N, dtype=np.int32)
    nxt = 0
    for i in range(N):
        r = find(i)
        if r not in root_map:
            root_map[r] = nxt
            nxt += 1
        labels[i] = root_map[r]
    return labels


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    shard_dir = os.path.join(WDS_ROOT, "matching_shards")
    all_shards = sorted(glob.glob(os.path.join(shard_dir, "shard-*.tar")))
    if not all_shards:
        print(f"No shards in {shard_dir}. Run prepare_matching_data.py first.")
        return

    # Train / val split by shard
    random.seed(42)
    random.shuffle(all_shards)
    n_val = max(1, int(len(all_shards) * args.val_split))
    val_shards, train_shards = all_shards[:n_val], all_shards[n_val:]
    print(f"Shards: {len(train_shards)} train, {len(val_shards)} val")

    train_ds = MatchingDataset(train_shards, augment=True)
    val_ds = MatchingDataset(val_shards, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, collate_fn=_matching_collate,
        num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, collate_fn=_matching_collate,
        num_workers=2,
    )

    model = PersonMatchingNetwork(
        feat_dim=args.feat_dim, num_heads=args.num_heads,
        num_layers=args.num_layers, roi_size=args.roi_size,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.save_dir, exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = _train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = _eval(model, val_loader, device)
        scheduler.step()

        print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            path = os.path.join(args.save_dir, "best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
            }, path)
            print(f"  Saved best → {path}  (acc={val_acc:.4f})")

    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "val_acc": val_acc,
    }, os.path.join(args.save_dir, "last.pt"))

    print(f"\nDone. Best val accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
