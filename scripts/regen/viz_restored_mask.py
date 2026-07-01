"""Quick 42-view mask viz for a restored swap24 seq.
Renders ~30 sampled frames as a 6×7 grid mp4, colored per person.
"""
import os
import sys
import cv2
import numpy as np
from os.path import join
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/simurgh/u/juze/code/HOIM3_Toolbox/scripts/utils")
from mask_io import SequenceShardReaders, load_frame_masks_shard_full

SEQ = sys.argv[1] if len(sys.argv) > 1 else "office_data32"
N_FRAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 30  # sampled at 1 fps (every 30 src frames)
OUT = sys.argv[3] if len(sys.argv) > 3 else f"/scr/juze/swap24_cleanup_logs/viz_restored_{SEQ}.mp4"

SHARD_ROOT = f"/simurgh2/datasets/HOI-M3/mask_shards/{SEQ}"
IMG_ROOT = f"/simurgh2/datasets/HOI-M3/images/{SEQ}"
VALIDITY_ROOT = f"/scr/juze/datasets/HOI-M3/mask_validity/{SEQ}"


def load_validity(frame_id):
    """Load per-(view, object) validity for a frame. Returns dict[obj] -> (42,) bool, or None if missing."""
    p = join(VALIDITY_ROOT, f"{frame_id:06d}.npz")
    if not os.path.isfile(p):
        return None
    try:
        d = np.load(p)
        out = {}
        for k in d.files:
            if k.endswith("_validity"):
                obj = k[:-len("_validity")]
                out[obj] = d[k].astype(bool)
        return out
    except Exception:
        return None


def is_view_invalid(validity_dict, view_idx):
    """A view is 'invalid' if ANY object has validity=0 for that view."""
    if validity_dict is None:
        return False
    for obj, v in validity_dict.items():
        if view_idx < len(v) and not v[view_idx]:
            return True
    return False

# Per-object colors (BGR) — persons stand out, objects in distinct hues
PERSON_COLORS = {
    "person0": (60, 60, 220),     # red
    "person1": (80, 220, 80),     # green
    "person2": (220, 200, 60),    # cyan-ish
    "person3": (220, 60, 220),    # magenta
}
# Bright BGR colors (saturated) — make objects pop against background.
OBJECT_COLORS = {
    "copier":          (  0, 165, 255),   # bright orange
    "displayer":       (  0, 255, 255),   # bright yellow
    "keyboard":        (255,   0, 255),   # bright magenta
    "officedesk":      (255, 150,   0),   # bright sky-blue
    "filebox":         (200,   0, 200),   # bright pink
    "toolkit":         (  0, 200, 255),   # bright gold
    "shredder":        (200,  50, 200),   # bright lavender
    "largetrashcan":   (255, 230,   0),   # bright cyan
    "bed":             (  0, 255, 100),   # bright lime
    "book":            (100, 255,   0),   # bright spring-green
    "cushion":         (100, 220, 255),   # bright peach
    "smallsofa":       (  0,   0, 255),   # bright red
    "television":      (100, 255, 100),   # bright light-green
    "radio":           (220,   0, 200),   # bright magenta-pink
    "airhumidifier":   ( 50, 220, 220),   # bright lemon
    "largestoragebox": (255, 100, 100),   # bright bright-blue
    "trashcan":        (180, 180, 255),   # bright salmon
    "bedside_cupboard":(  0, 180, 255),   # bright orange-yellow
    "cabinet":         (200, 255,   0),   # bright neon-green
    "lamp2":           (255,   0, 100),   # bright violet
    "woodenchair":     (100, 255, 200),   # bright pale-green
    "coffeemachine":   ( 80, 255,  80),   # bright light-green
    "officechair":     (255, 180, 220),   # bright pale-purple
}

def overlay_view(img: np.ndarray, masks: dict, view_idx: int, alpha_person=0.55, alpha_obj=0.55, invalid: bool = False) -> np.ndarray:
    """Overlay person + object masks on one view's image. masks[obj] is (V, H, W) uint8.

    Objects are drawn first (lower alpha), persons on top (higher alpha) so persons stay visible.
    """
    h, w = img.shape[:2]
    out = img.copy().astype(np.float32)

    def draw(obj, color, alpha):
        full_arr = masks.get(obj)
        if full_arr is None:
            return
        m = full_arr[view_idx]
        if m.shape[:2] != (h, w):
            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        sel = m.astype(bool)
        if sel.sum() == 0:
            return
        color = np.array(color, dtype=np.float32)
        out[sel] = out[sel] * (1 - alpha) + color * alpha

    # Objects first (lower alpha so persons stay visible)
    for obj in sorted(masks.keys()):
        if obj.startswith("person"):
            continue
        color = OBJECT_COLORS.get(obj, (180, 180, 180))
        draw(obj, color, alpha_obj)
    # Persons on top
    for obj in masks.keys():
        if not obj.startswith("person"):
            continue
        color = PERSON_COLORS.get(obj, (180, 180, 180))
        draw(obj, color, alpha_person)

    out = out.clip(0, 255).astype(np.uint8)
    # If invalid view, overlay red border + "INVALID" label
    if invalid:
        h, w = out.shape[:2]
        border = 8
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), (0, 0, 255), border)
        cv2.putText(out, "INVALID", (w // 2 - 90, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.putText(out, f"v{view_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return out


def render_one_frame(seq_readers, frame_id, n_views, mask_h, mask_w, img_h, img_w):
    masks = load_frame_masks_shard_full(seq_readers, frame_id)
    tiles = []
    for v in range(n_views):
        img_path = join(IMG_ROOT, str(v), f"{frame_id:06d}.jpg")
        if not os.path.exists(img_path):
            tile = np.zeros((img_h, img_w, 3), dtype=np.uint8)
            cv2.putText(tile, f"v{v} MISS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (60, 60, 60), 2)
        else:
            img = cv2.imread(img_path)
            if img is None or img.shape[:2] != (img_h, img_w):
                img = cv2.resize(img, (img_w, img_h)) if img is not None else np.zeros((img_h, img_w, 3), dtype=np.uint8)
            tile = overlay_view(img, masks, v)
        tiles.append(tile)
    # 6 rows × 7 cols = 42
    rows = []
    for r in range(6):
        row = np.hstack(tiles[r * 7:(r + 1) * 7])
        rows.append(row)
    return np.vstack(rows)


def main():
    print(f"[viz] seq={SEQ}, shard={SHARD_ROOT}, out={OUT}")
    sr = SequenceShardReaders(SHARD_ROOT)
    print(f"  objects={sr.objects}, views={sr.views}, frames={sr.meta['num_frames']}, mask_size={sr.height}x{sr.width}")
    persons = [o for o in sr.objects if o.startswith("person")]
    print(f"  persons={persons}")

    # Image resolution: 720p
    img_h, img_w = 720, 1280
    # tile output: 6 rows × 7 cols ; each tile downsampled to e.g. 320x180 (a 16:9 thumbnail)
    tile_w, tile_h = 320, 180

    total = sr.meta["num_frames"]
    sample_ids = np.linspace(0, total - 1, N_FRAMES, dtype=int).tolist()
    print(f"  rendering {len(sample_ids)} sampled frames")

    h_grid = 6 * tile_h
    w_grid = 7 * tile_w
    writer = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), 1, (w_grid, h_grid))

    for i, fid in enumerate(sample_ids):
        try:
            masks = load_frame_masks_shard_full(sr, fid)
        except KeyError:
            print(f"  frame {fid}: missing, skip")
            continue
        validity = load_validity(fid)  # None if mask_validity not present
        tiles = []
        for v in range(sr.views):
            img_path = join(IMG_ROOT, str(v), f"{fid:06d}.jpg")
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is None:
                    img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
            else:
                img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
                cv2.putText(img, "no img", (200, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, (80, 80, 80), 3)
            invalid = is_view_invalid(validity, v)
            tile = overlay_view(img, masks, v, invalid=invalid)
            tile = cv2.resize(tile, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            tiles.append(tile)
        rows = []
        for r in range(6):
            rows.append(np.hstack(tiles[r * 7:(r + 1) * 7]))
        grid = np.vstack(rows)
        # Count invalid views for banner
        n_invalid = sum(1 for v in range(sr.views) if is_view_invalid(validity, v)) if validity is not None else None
        # Top banner
        banner = np.zeros((30, w_grid, 3), dtype=np.uint8)
        v_txt = f"invalid={n_invalid}/42" if n_invalid is not None else "validity=N/A"
        cv2.putText(banner, f"{SEQ}  frame={fid}  persons={','.join(persons)}  {v_txt}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
        grid = np.vstack([banner, grid])
        # Re-init writer with correct height on first frame
        if i == 0:
            writer.release()
            writer = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), 1, (grid.shape[1], grid.shape[0]))
        writer.write(grid)
        if (i + 1) % 5 == 0:
            print(f"  rendered {i+1}/{len(sample_ids)} (frame={fid})")

    writer.release()
    sr.close()
    print(f"[viz] DONE → {OUT}")


if __name__ == "__main__":
    main()
