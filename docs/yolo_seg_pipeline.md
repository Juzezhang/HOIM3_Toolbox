# YOLO11-seg + Cross-view Person Matching Pipeline

A two-stage pipeline for HOI-M3 multi-view data:
1. **YOLO11-seg Instance Segmentation**: Train on sequences with mask annotations, infer on sequences without masks.
2. **Cross-view Person Matching Network**: Leverage consistent person IDs from mask data to train a person identity matching model across 42 camera views.

---

## 1. Data Overview

| Item | Value |
|------|-------|
| Training sequences (with masks) | 190 |
| Test sequences (no masks) | 14 |
| Frames per sequence | ~21,600 (59.94 fps) |
| Camera views | 42 |
| Video resolution | 3840x2160 (4K HEVC) |
| Mask resolution | 1080x1920 |
| Number of classes | 88 (person + 87 objects) |
| Sampling step | step=60 -> ~360 frames/seq |
| Total training images | 190 x 360 x 42 = **~2.9M** |

### Data Paths

```
Source data:    /simurgh/group/juze/datasets/HOI-M3/
Mask shards:    /simurgh2/datasets/HOI-M3/mask_shards/
Output root:    /simurgh2/datasets/HOI-M3/yolo_seg_wds/
```

---

## 2. Pipeline Architecture

```
+------------------------------------------------------------------+
|                      Stage 1: YOLO11-seg                          |
|                                                                   |
|  Step 1                Step 2                Step 3               |
|  prepare_webdataset    train                 predict_visualize    |
|  ------------------>   ------------------>   ------------------>  |
|  mask shards           WebDataset shards     trained model        |
|  + 4K videos     ->    + val set       ->    fine-tuned     ->    |
|  -> .tar shards        YOLO11m-seg.pt        test predictions     |
|                                                                   |
+-------------------------------------------------------------------+
|                   Stage 2: Cross-view Matching                    |
|                                                                   |
|  Step 4                          Step 5                           |
|  prepare_matching_data           train_matching                   |
|  -------------------------->     -------------------------->      |
|  mask shards + videos      ->    matching shards            ->    |
|  -> composite grid images        ResNet50+Transformer model       |
|     with person bboxes           -> pairwise affinity matrix      |
+-------------------------------------------------------------------+
```

---

## 3. Train/Test Split and Nested mask_npz Fix

### Bug Fix: Nested mask_npz Directories

36 sequences had their mask NPZ files extracted into a nested directory:
```
mask_npz/{seq_name}/{seq_name}/*.npz    # wrong (double-nested)
```
instead of the expected:
```
mask_npz/{seq_name}/*.npz               # correct (flat)
```

This caused these sequences to be misclassified as having no mask data. The fix was to move all files up one level (`os.rename` from nested to parent, then `rmdir` the empty nested directory). After the fix, all 87 object classes are covered in the training set with **zero unseen objects** in the test set.

### Final Split

| Set | Sequences | Notes |
|-----|-----------|-------|
| **Train** | 190 | All have mask shards (154 original + 36 fixed) |
| **Test** | 14 | No mask data at all |

**Test sequences (14):**
```
bedroom_data34*    diningroom_data06  livingroom_data03  livingroom_data04
livingroom_data07  livingroom_data11  livingroom_data30  livingroom_data31
livingroom_data32  livingroom_data33  livingroom_data34  office_data03
office_data18      office_data30      office_data61
```

> `*` bedroom_data34 has only 10 NPZ frames (likely incomplete extraction); it is included in training but contributes minimal data.

---

## 4. Class Mapping

All `personN` variants (person0, person1, ...) are merged into **class 0: "person"**. The 87 unique object names are sorted alphabetically and assigned class IDs 1-87:

```
0: person
1: airhumidifier
2: ashbucket
3: banana
4: barbell
5: baseballbat
6: basketball
7: bed
...
87: yogamat
```

The mapping is saved to `class_mapping.json` and the YOLO-format `data.yaml`.

---

## 5. Script Details

### 5.0 `config.py` -- Shared Constants

Paths, constants, and utility functions shared by all scripts:

- **Path constants**: `DATA_ROOT`, `SHARD_ROOT`, `VALIDITY_ROOT`, `VIDEO_ROOT`, `WDS_ROOT`
- **Data constants**: `MASK_H=1080`, `MASK_W=1920`, `NUM_VIEWS=42`, `FRAME_STEP=60`, `IMGSZ=640`
- **Name corrections**: Fixes known typos in mask data (e.g., `matermelon` -> `watermelon`)
- **Class mapping builder**: `build_class_mapping()` collects all object names from `sequence_contents.json`
- **Sequence partitioning**: `get_train_sequences()` (have mask shards) / `get_test_sequences()` (no masks)

---

### 5.1 `prepare_webdataset.py` -- Data Preparation (Step 1)

**Core script.** Converts mask shards + 4K videos into WebDataset `.tar` shards + a YOLO-format validation set.

#### Algorithm

```
For each training sequence (154 total):
  1. Open SequenceShardReaders (mask shard reader)
  2. Determine target frame list (step=60 -> ~364 frames)
  3. Build object -> class_id mapping

  4. [Parallel] Preload video frames:
     For each view (42 total), using ThreadPoolExecutor(workers=8):
       Launch ffmpeg subprocess to decode 4K HEVC video
       ffmpeg flags: -skip_frame nokey -flags2 +fast (fast decode path)
       Stream 640x360 rawvideo -> Python selects every step-th frame -> JPEG encode

  5. Frame-by-frame processing:
     For each target frame:
       a. Load all object masks from shard: {obj: (42, 1080, 1920) uint8}
       b. Load mask_validity NPZ -> per-view per-object validity flags

       For each view (0-41):
         c. Retrieve preloaded JPEG image
         d. For each object:
            - Check validity (skip if invalid)
            - Binarize mask: mask[view] > 127
            - Bbox: cv2.boundingRect -> normalized [cx, cy, w, h]
            - Polygon: cv2.findContours(RETR_EXTERNAL, CHAIN_APPROX_TC89_L1)
              Filter contours with area < 100px, normalize coords to [0, 1]
            - Map class: personN -> 0, object -> 1-87

         e. If valid annotations exist -> write sample to WebDataset shard
         f. If validation frame -> also write to YOLO directory format
```

#### WebDataset Shard Format

Each `.tar` file contains ~1000 samples:
```
{seq}_{frame:06d}_{view:02d}.jpg     # RGB image 640x360 JPEG
{seq}_{frame:06d}_{view:02d}.json    # annotation JSON
```

Annotation JSON:
```json
{
  "img_w": 1920, "img_h": 1080,
  "annotations": [
    {
      "class_id": 0,
      "bbox": [0.5, 0.5, 0.2, 0.3],
      "segments": [[[0.4, 0.35], [0.6, 0.35], [0.6, 0.65], [0.4, 0.65]]]
    }
  ]
}
```

All coordinates are normalized to [0, 1] (YOLO convention).

#### Validation Set

Every 5th training sequence, frame step=300, written in standard YOLO directory format:
```
val/images/{key}.jpg
val/labels/{key}.txt    # Format: class_id px1 py1 px2 py2 ... pNx pNy
```

#### Performance Optimizations

| Optimization | Effect |
|-------------|--------|
| ffmpeg pipe instead of OpenCV | 4x faster 4K HEVC decode (7.5 min vs 29 min/seq) |
| `-skip_frame nokey -flags2 +fast` | Enables fast decode path, skips deblocking filter |
| ThreadPoolExecutor(8) | 8-way parallel video decoding |
| On-the-fly JPEG encoding | Memory holds compressed bytes only, not raw pixels |
| LZ4 parallel decompression | 7 object masks decompressed concurrently per frame |

#### Usage

```bash
python scripts/yolo_seg/prepare_webdataset.py --step 60 --imgsz 640 --workers 8
# Optional: process a subset of sequences
python scripts/yolo_seg/prepare_webdataset.py --sequences bedroom_data01 bedroom_data02
```

#### Expected Output

- ~2.3M samples x ~50KB JPEG = **~115 GB** of tar shards
- ~2,300 tar files
- Validation set: ~46K images/labels

---

### 5.2 `train.py` -- YOLO11-seg Fine-tuning (Step 2)

#### Core Design: Subclassing SegmentationTrainer

```python
class WDSSegTrainer(SegmentationTrainer):
    def get_dataloader(self, dataset_path, batch_size, rank, mode):
        if mode == "train":
            return build_wds_dataloader(...)  # WebDataset streaming
        return super().get_dataloader(...)     # Standard YOLO format for val
```

Training uses WebDataset streaming; validation uses standard YOLO directory format.

#### Data Processing Pipeline

```
WebDataset .tar -> Decode JPEG -> Letterbox 640x640 -> Augment -> Rasterize masks
```

**Letterbox padding** (16:9 -> square):
- 640x360 image -> 640x640 canvas with 140px gray padding top and bottom
- Coordinate adjustment: `y_new = y_old * 0.5625 + 0.21875`, `x` unchanged

**Data augmentation** (V1, no mosaic):
- Horizontal flip (p=0.5)
- HSV jitter (hue=0.015, sat=0.7, val=0.4)

**Mask rasterization**:
- Normalized polygon points are rasterized into 160x160 binary masks (imgsz/4)
- `cv2.fillPoly` draws polygons in letterbox coordinate space

#### Batch Format

Compatible with ultralytics loss functions:
```python
batch = {
    "img":       (B, 3, 640, 640)    # uint8, divided by 255 during training
    "cls":       (N_total, 1)         # class indices
    "bboxes":    (N_total, 4)         # xywh normalized
    "masks":     (N_total, 160, 160)  # binary masks
    "batch_idx": (N_total,)           # image index per instance
}
```

#### Training Configuration

```
Model:    YOLO11m-seg (medium -- balances speed and accuracy for 88 classes)
Epochs:   50
Batch:    32
LR:       0.001
Device:   GPU 0 (multi-GPU: --device 0,1)
```

#### Usage

```bash
python scripts/yolo_seg/train.py \
    --data /simurgh2/datasets/HOI-M3/yolo_seg_wds/data.yaml \
    --model yolo11m-seg.pt --epochs 50 --batch 32 --device 0
```

---

### 5.3 `predict_visualize.py` -- Inference and Visualization (Step 3)

Runs the trained model on the 50 test sequences that have no mask annotations:

```
For each test sequence:
  For selected views (e.g., 0, 7, 14, 21, 28, 35):
    1. Preload video frames (step=60, 640x360)
    2. model.predict(frame, conf=0.25)
    3. results[0].plot() -> draw bbox + class label + segmentation mask
    4. Save annotated frames as JPEG
    5. Compile into MP4 video
```

#### Output Structure

```
predictions/
  {seq_name}/
    view_{v}/
      frame_{f:06d}.jpg     # annotated frames
    view_{v}.mp4             # compiled video
```

#### Usage

```bash
python scripts/yolo_seg/predict_visualize.py \
    --model /simurgh2/datasets/HOI-M3/yolo_seg_wds/runs/yolo11m-seg/weights/best.pt \
    --views 0 7 14 21 28 35 --step 60 --conf 0.25
```

---

### 5.4 `prepare_matching_data.py` -- Matching Data Preparation (Step 4)

#### Objective

Leverage the natural supervision signal in mask data: `person0.shard` contains the **same physical person** across all 42 views. This provides ground-truth cross-view correspondences.

#### Algorithm

```
For each training frame (step=60):
  1. Load all person masks -> determine which views each person is valid in
  2. Compute the union of all valid views
  3. Generate `augment` different view combinations:
     a. Stratified sampling of K=6 views (evenly spread across 42 cameras)
     b. Build 2x3 grid composite image (960x360):
        Each view at 320x180, arranged in the grid
     c. For each person visible in >=2 selected views:
        - Compute bbox from mask (cv2.boundingRect)
        - Scale to view resolution -> offset by grid position -> normalize
        - Record: person_id, view_idx, grid_pos, bbox
     d. Write to WebDataset shard
```

#### Composite Image and Annotation Format

```
+----------+----------+----------+
|  View 0  |  View 7  | View 14  |   Each cell 320x180
+----------+----------+----------+
| View 21  | View 28  | View 35  |   Composite 960x360
+----------+----------+----------+
```

Annotation JSON:
```json
{
  "views": [0, 7, 14, 21, 28, 35],
  "grid_rows": 2, "grid_cols": 3,
  "persons": [
    {
      "person_id": 0,
      "detections": [
        {"view_idx": 0,  "grid_pos": 0, "bbox": [cx, cy, w, h]},
        {"view_idx": 14, "grid_pos": 2, "bbox": [cx, cy, w, h]}
      ]
    }
  ]
}
```

#### View Sampling Strategy

Stratified sampling ensures diverse camera angles: the 42 cameras are divided into K bins, and one camera is randomly selected from each bin.

#### Data Scale

- 154 sequences x ~360 frames x 3 augmentations = **~166K composite samples**

#### Usage

```bash
python scripts/yolo_seg/prepare_matching_data.py \
    --step 60 --num_views 6 --augment 3 --workers 8
```

---

### 5.5 `train_matching.py` -- Cross-view Person Matching Network (Step 5)

#### Network Architecture

```
Input:
  composite image (K views in grid)  ----------+
  N person bbox coordinates          ------+   |
  N view indices                     ---+  |   |
                                        |  |   |
  +-------------------------------------+--+---+
  | ResNet50 Backbone                   |  |
  | (freeze stages 1-2)                |  |
  | -> feature map (1, 2048, H/32, W/32)  |
  +------------------+------------------+  |
                     |                     |
  +------------------+---------------------+
  | RoIAlign (7x7)   |
  | -> per-person features (N, 2048, 7, 7)
  +------------------+---------------------+
                     |                     |
  +------------------+                     |
  | Linear Projection                     |
  | (2048x7x7 -> 256)                     |
  | + View Positional Encoding (Embedding) <+
  | -> (N, 256)
  +------------------+---------------------+
                     |                     |
  +------------------+                     |
  | Transformer Encoder                   |
  | (4 layers, 8 heads, dim=256)          |
  | Cross-attention across all detections |
  | -> (N, 256)                           |
  +------------------+---------------------+
                     |
  +------------------+
  | Affinity Head
  | Linear(256 -> 256) -> proj
  | affinity = sigmoid(proj @ proj.T)
  | -> (N, N) pairwise similarity matrix
  +------------------------------------------
```

#### Design Rationale

| Design Choice | Rationale |
|--------------|-----------|
| Composite grid image | Enables the network to use both **appearance** and **spatial** cues |
| RoIAlign | Precisely extracts per-person features from the backbone feature map using YOLO-detected bboxes |
| View Embedding | Informs the network which camera angle each detection originates from |
| Transformer cross-attention | Naturally handles the set matching problem; supports variable numbers of persons and views |
| Affinity matrix | More flexible than direct classification; supports arbitrary numbers of identities |

#### Loss Function

- **Binary Cross-Entropy** on the NxN affinity matrix
- Ground truth: same `person_id` -> 1, different -> 0

#### Inference Pipeline

```
1. YOLO detects person bboxes in all views
2. Select K views, build composite image
3. Matching network -> affinity matrix
4. Threshold (>0.5) -> adjacency graph
5. Connected components (Union-Find) -> consistent person IDs
6. Propagate IDs to remaining views via transitivity
```

#### Training Configuration

```
Backbone:    ResNet50 (ImageNet pretrained, stages 1-2 frozen)
ROI size:    7x7
Transformer: 4 layers, 8 heads, dim=256
Optimizer:   AdamW, lr=1e-4, weight_decay=1e-4
Scheduler:   CosineAnnealing
Batch:       32
Epochs:      30
```

#### Usage

```bash
python scripts/yolo_seg/train_matching.py \
    --epochs 30 --batch 32 --lr 1e-4 --device 0
```

---

## 6. Key Dependency Modules

This pipeline reuses existing modules from the project:

| Module | Source | Purpose |
|--------|--------|---------|
| `SequenceShardReaders` | `scripts/utils/mask_io.py` | Read LZ4+bitpack mask shards |
| `load_frame_masks_shard_full()` | `scripts/utils/mask_io.py` | Parallel loading of all object masks for one frame |
| `preload_video_frames()` | `scripts/visualize_mask_validity.py` | Efficient video frame extraction (val set / matching data) |
| `resolve_view_validity()` | `scripts/visualize_mask_validity.py` | Determine whether a mask is valid for a specific view |
| `NAME_CORRECTIONS` | `scripts/extract_sequence_contents.py` | Fix known mask name typos |

---

## 7. File Structure

```
scripts/yolo_seg/
|-- __init__.py                  # Package marker
|-- config.py                    # Shared paths, constants, class mapping
|-- prepare_webdataset.py        # Step 1: masks -> bbox + polygon -> WebDataset
|-- train.py                     # Step 2: YOLO11-seg fine-tuning
|-- predict_visualize.py         # Step 3: Test sequence inference + visualization
|-- prepare_matching_data.py     # Step 4: Cross-view composite images + person bboxes
|-- train_matching.py            # Step 5: ResNet50 + Transformer matching network
`-- run_pipeline.sh              # One-command full pipeline runner
```

Output structure:
```
/simurgh2/datasets/HOI-M3/yolo_seg_wds/
|-- class_mapping.json           # {class_to_id, class_names}
|-- data.yaml                    # YOLO data config
|-- train_shards/                # ~2,300 .tar files
|   |-- shard-000000.tar
|   `-- ...
|-- val/                         # YOLO-format validation set
|   |-- images/
|   `-- labels/
|-- runs/                        # YOLO training output
|   `-- yolo11m-seg/
|       `-- weights/best.pt
|-- predictions/                 # Test inference results
|   `-- {seq_name}/view_{v}.mp4
|-- matching_shards/             # Matching training data
|   `-- shard-*.tar
|-- matching_runs/               # Matching network weights
|   |-- best.pt
|   `-- last.pt
`-- logs/                        # Per-step logs
```

---

## 8. Usage Guide

### Prerequisites

```bash
conda activate HOIM3_Toolbox   # or your current Python environment
pip install webdataset          # streaming data loading
# ultralytics is already installed (8.3.162)
```

### One-Command Run

```bash
# Launch Step 1 in background
nohup python3 scripts/yolo_seg/prepare_webdataset.py --step 60 --workers 8 \
  > /simurgh2/datasets/HOI-M3/yolo_seg_wds/prepare_webdataset.log 2>&1 &

# Launch pipeline (waits for Step 1, then auto-runs Steps 2-5)
nohup bash scripts/yolo_seg/run_pipeline.sh \
  > /simurgh2/datasets/HOI-M3/yolo_seg_wds/logs/pipeline.log 2>&1 &
```

### Step-by-Step

```bash
# Step 1
python scripts/yolo_seg/prepare_webdataset.py --step 60 --workers 8

# Step 2
python scripts/yolo_seg/train.py --epochs 50 --batch 32 --device 0

# Step 3
python scripts/yolo_seg/predict_visualize.py --model .../best.pt --views 0 7 14 21 28 35

# Step 4
python scripts/yolo_seg/prepare_matching_data.py --step 60 --num_views 6 --augment 3

# Step 5
python scripts/yolo_seg/train_matching.py --epochs 30 --batch 32 --device 0
```

### Monitoring

```bash
# Step 1 progress
grep "samples" /simurgh2/datasets/HOI-M3/yolo_seg_wds/prepare_webdataset.log | wc -l  # sequences completed
du -sh /simurgh2/datasets/HOI-M3/yolo_seg_wds/train_shards/                            # total size

# Pipeline status
tail -20 /simurgh2/datasets/HOI-M3/yolo_seg_wds/logs/pipeline.log

# YOLO training (Step 2)
tail -20 /simurgh2/datasets/HOI-M3/yolo_seg_wds/logs/train.log
```

---

## 9. Estimated Runtime

| Step | Estimated Duration |
|------|--------------------|
| Step 1: Data preparation | ~13 hours (7.5 min/seq x 154) |
| Step 2: YOLO training | ~12-24 hours (50 epochs) |
| Step 3: Inference + visualization | ~1 hour |
| Step 4: Matching data preparation | ~10 hours |
| Step 5: Matching network training | ~2-4 hours |
| **Total** | **~2-3 days** |
