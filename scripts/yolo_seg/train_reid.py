#!/usr/bin/env python3
"""Train cross-view person ReID model for consistent person ID assignment.

Architecture: ResNet50 backbone → embedding → triplet loss + cross-entropy
At inference: extract embeddings for person crops across views, cluster by cosine similarity.

Usage:
    python scripts/yolo_seg/train_reid.py --data /simurgh2/datasets/HOI-M3/reid_data --device 1
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm


# ── Dataset ──────────────────────────────────────────────────────────────────

class ReIDDataset(Dataset):
    """Person ReID dataset with triplet sampling.

    Each sample returns (anchor, positive, negative) where:
    - anchor: person crop from one view
    - positive: same person from a different view
    - negative: different person from any view
    """

    def __init__(self, root_dir, transform=None, is_train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train

        # Build index: pid -> list of image paths
        self.pid_to_images = defaultdict(list)
        self.all_images = []
        self.all_labels = []
        self.pid_to_label = {}

        for label_idx, pid in enumerate(sorted(os.listdir(root_dir))):
            pid_dir = os.path.join(root_dir, pid)
            if not os.path.isdir(pid_dir):
                continue
            self.pid_to_label[pid] = label_idx
            images = sorted([
                os.path.join(pid_dir, f)
                for f in os.listdir(pid_dir)
                if f.endswith('.jpg')
            ])
            self.pid_to_images[pid] = images
            for img_path in images:
                self.all_images.append(img_path)
                self.all_labels.append(label_idx)

        self.pids = list(self.pid_to_images.keys())
        self.num_classes = len(self.pids)
        print(f"  {root_dir}: {len(self.all_images)} images, {self.num_classes} identities")

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        anchor_path = self.all_images[idx]
        anchor_label = self.all_labels[idx]
        anchor_pid = self.pids[anchor_label]

        # Load anchor
        anchor = self._load_image(anchor_path)

        if self.is_train:
            # Positive: same person, different image (ideally different view)
            pos_images = self.pid_to_images[anchor_pid]
            pos_path = random.choice([p for p in pos_images if p != anchor_path] or pos_images)
            positive = self._load_image(pos_path)

            # Negative: different person
            neg_pid = random.choice([p for p in self.pids if p != anchor_pid])
            neg_path = random.choice(self.pid_to_images[neg_pid])
            negative = self._load_image(neg_path)

            return anchor, positive, negative, anchor_label
        else:
            return anchor, anchor_label

    def _load_image(self, path):
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((256, 128, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        return img


# ── Model ────────────────────────────────────────────────────────────────────

class ReIDModel(nn.Module):
    """ResNet50-based ReID model with embedding + classification heads."""

    def __init__(self, num_classes, embed_dim=256):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        # Remove final FC
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embed = nn.Sequential(
            nn.Linear(2048, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.pool(feat).flatten(1)  # (B, 2048)
        emb = self.embed(feat)  # (B, embed_dim)
        emb_norm = F.normalize(emb, p=2, dim=1)
        logits = self.classifier(emb)
        return emb_norm, logits


# ── Losses ───────────────────────────────────────────────────────────────────

class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        dist_ap = (anchor - positive).pow(2).sum(1)
        dist_an = (anchor - negative).pow(2).sum(1)
        loss = F.relu(dist_ap - dist_an + self.margin)
        return loss.mean()


# ── Training ─────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, triplet_loss, ce_loss, device):
    model.train()
    total_loss = 0
    total_tri = 0
    total_ce = 0
    n = 0

    for anchor, positive, negative, labels in tqdm(loader, desc="  train", leave=False):
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        labels = labels.to(device)

        emb_a, logits_a = model(anchor)
        emb_p, _ = model(positive)
        emb_n, _ = model(negative)

        loss_tri = triplet_loss(emb_a, emb_p, emb_n)
        loss_ce = ce_loss(logits_a, labels)
        loss = loss_tri + loss_ce

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total_loss += loss.item()
        total_tri += loss_tri.item()
        total_ce += loss_ce.item()
        n += 1

    return total_loss / n, total_tri / n, total_ce / n


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate: compute embeddings and measure retrieval accuracy."""
    model.eval()
    all_embs = []
    all_labels = []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        emb, _ = model(imgs)
        all_embs.append(emb.cpu())
        all_labels.append(labels)

    embs = torch.cat(all_embs)  # (N, dim)
    labels = torch.cat(all_labels)  # (N,)

    # Compute pairwise cosine similarity
    sim = embs @ embs.T  # (N, N)

    # For each query, check if nearest neighbor has same label
    correct = 0
    total = 0
    for i in range(len(embs)):
        sim[i, i] = -1  # exclude self
        nn_idx = sim[i].argmax()
        if labels[nn_idx] == labels[i]:
            correct += 1
        total += 1

    rank1_acc = correct / max(total, 1)
    return rank1_acc


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Cross-view person ReID training")
    p.add_argument("--data", type=str, default="/simurgh2/datasets/HOI-M3/reid_data")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--device", type=str, default="1")
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument("--resume", type=str, default=None,
                   help="Resume from checkpoint (path to last_reid.pt)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.save_dir is None:
        args.save_dir = os.path.join(args.data, "runs")
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop((256, 128), padding=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Datasets
    print("Loading datasets...")
    train_ds = ReIDDataset(os.path.join(args.data, "train"), train_transform, is_train=True)
    val_ds = ReIDDataset(os.path.join(args.data, "val"), val_transform, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch * 2, shuffle=False,
                            num_workers=4, pin_memory=True)

    # Model
    model = ReIDModel(num_classes=train_ds.num_classes, embed_dim=args.embed_dim).to(device)
    print(f"Model: ResNet50 → {args.embed_dim}d embedding → {train_ds.num_classes} classes")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    triplet_loss = TripletLoss(margin=0.3)
    ce_loss = nn.CrossEntropyLoss()

    best_acc = 0.0
    start_epoch = 1
    results = []

    # Resume from checkpoint
    resume_path = args.resume
    if resume_path is None:
        # Auto-detect: resume from last_reid.pt if it exists
        auto_path = os.path.join(args.save_dir, "last_reid.pt")
        if os.path.isfile(auto_path):
            resume_path = auto_path

    if resume_path and os.path.isfile(resume_path):
        print(f"Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc = ckpt.get("best_rank1", ckpt.get("rank1", 0.0))
        if "results" in ckpt:
            results = ckpt["results"]
        print(f"  Resumed at epoch {start_epoch}, best rank1={best_acc:.4f}")

    for epoch in range(start_epoch, args.epochs + 1):
        loss, tri, ce = train_one_epoch(model, train_loader, optimizer, triplet_loss, ce_loss, device)
        rank1 = evaluate(model, val_loader, device)
        scheduler.step()

        print(f"Epoch {epoch}/{args.epochs}: loss={loss:.4f} (tri={tri:.4f} ce={ce:.4f}) rank1={rank1:.4f}")
        results.append({"epoch": epoch, "loss": loss, "tri": tri, "ce": ce, "rank1": rank1})

        # Save checkpoint with full state for resume
        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "rank1": rank1,
            "best_rank1": best_acc,
            "results": results,
        }

        if rank1 > best_acc:
            best_acc = rank1
            ckpt_data["best_rank1"] = best_acc
            torch.save(ckpt_data, os.path.join(args.save_dir, "best_reid.pt"))
            print(f"  Saved best (rank1={rank1:.4f})")

        # Always save last
        torch.save(ckpt_data, os.path.join(args.save_dir, "last_reid.pt"))

    # Save results
    with open(os.path.join(args.save_dir, "reid_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone! Best rank-1 accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
