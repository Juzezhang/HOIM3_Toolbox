"""Probe SAM3 text-prompt detection for missing-mask objects in office_data32/33.

For each (seq, object, view) combination, run SAM3 image segmentation with the
object name as text prompt, save preview PNG with bbox + mask overlay.

Usage: python sam3_detect_missing_obj.py
"""
import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Configure
IMG_ROOT = "/simurgh2/datasets/HOI-M3/images"
OUT_DIR = "/simurgh2/datasets/HOI-M3/sam3_missing_obj_probe"
os.makedirs(OUT_DIR, exist_ok=True)

# Targets: (seq, object, candidate_views, candidate_frames)
# Pick spread-out views + frames so we have variety to inspect
TARGETS = [
    ("office_data32", "shredder", [0, 7, 14, 21, 28, 35], [0, 5000, 10000]),
    ("office_data33", "radio",    [0, 7, 14, 21, 28, 35], [0, 5000, 10000]),
]

# Load SAM3 once
print("[SAM3] Loading model...")
from PIL import Image
import torch
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import plot_results

model = build_sam3_image_model(load_from_HF=True)
processor = Sam3Processor(model)
print("[SAM3] Model loaded.")


def draw_overlay(img_bgr, mask, color=(0, 255, 0), alpha=0.5):
    """Draw mask as semi-transparent overlay."""
    out = img_bgr.copy().astype(np.float32)
    sel = mask.astype(bool)
    if sel.sum() > 0:
        out[sel] = out[sel] * (1 - alpha) + np.array(color, dtype=np.float32) * alpha
    return out.clip(0, 255).astype(np.uint8)


for seq, obj_name, views, frames in TARGETS:
    print(f"\n=== {seq}: detect '{obj_name}' ===")
    for fid in frames:
        for view in views:
            img_path = os.path.join(IMG_ROOT, seq, str(view), f"{fid:06d}.jpg")
            if not os.path.isfile(img_path):
                continue
            try:
                pil = Image.open(img_path).convert("RGB")
                state = processor.set_image(pil)
                out = processor.set_text_prompt(state=state, prompt=obj_name)
                masks = out.get("masks", [])
                scores = out.get("scores", [])
                boxes = out.get("boxes", [])
                if hasattr(masks, "cpu"):
                    masks = masks.cpu().numpy()
                if hasattr(scores, "cpu"):
                    scores = scores.cpu().numpy()
                if hasattr(boxes, "cpu"):
                    boxes = boxes.cpu().numpy()
                n = len(scores) if scores is not None else 0
                print(f"  v{view:02d} f{fid:06d}: {n} detection(s)", end="")
                if n > 0:
                    print(f", top score={scores[0]:.3f}")
                    # Save overlay
                    img_bgr = cv2.imread(img_path)
                    h, w = img_bgr.shape[:2]
                    # Pick top mask — handle (N,H,W) / (1,N,H,W) / (H,W) shapes
                    m = np.asarray(masks[0])
                    while m.ndim > 2:
                        m = m[0]
                    # if mask shape doesn't match img, resize
                    if m.shape != (h, w):
                        m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                    overlay = draw_overlay(img_bgr, m, color=(0, 255, 0), alpha=0.5)
                    # Draw bbox (overlay all detections too)
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = box.astype(int)
                        col = (0, 255, 255) if i == 0 else (0, 200, 255)
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), col, 2)
                        if i < len(scores):
                            cv2.putText(overlay, f"{scores[i]:.2f}", (x1, max(y1-5, 15)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
                    cv2.putText(overlay,
                                f"{seq} {obj_name} v{view} f{fid} top={scores[0]:.2f} n={n}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 255, 255), 2)
                    out_path = os.path.join(
                        OUT_DIR,
                        f"{seq}__{obj_name}__v{view:02d}__f{fid:06d}.png"
                    )
                    cv2.imwrite(out_path, overlay)
                else:
                    print()
            except Exception as e:
                print(f"  v{view:02d} f{fid:06d}: ERROR {e}")

print(f"\n[done] previews saved to {OUT_DIR}")
