#!/bin/bash
# Full YOLO-seg + matching pipeline runner.
# Waits for Step 1 (prepare_webdataset) to finish, then runs Steps 2-5.
#
# Usage:
#   bash scripts/yolo_seg/run_pipeline.sh            # wait for running Step 1
#   bash scripts/yolo_seg/run_pipeline.sh --from 2   # skip Step 1, start from Step 2

set -e

WDS_ROOT="/simurgh2/datasets/HOI-M3/yolo_seg_wds"
LOG_DIR="$WDS_ROOT/logs"
mkdir -p "$LOG_DIR"

FROM_STEP=${1:-1}
if [[ "$1" == "--from" ]]; then
    FROM_STEP=$2
fi

echo "=== HOI-M3 YOLO-seg Pipeline ==="
echo "Starting from step $FROM_STEP"
echo ""

# ── Step 1: Wait for data preparation ────────────────────────────────────────
if [[ $FROM_STEP -le 1 ]]; then
    echo "[Step 1] Waiting for prepare_webdataset.py to finish..."
    # Wait for the nohup process
    while pgrep -f "prepare_webdataset.py" > /dev/null 2>&1; do
        SHARDS=$(ls "$WDS_ROOT/train_shards/" 2>/dev/null | wc -l)
        echo "  $(date '+%H:%M:%S') - $SHARDS shards written"
        sleep 300
    done
    echo "[Step 1] DONE. $(ls $WDS_ROOT/train_shards/ | wc -l) shards, $(ls $WDS_ROOT/val/images/ 2>/dev/null | wc -l) val images"
fi

# ── Step 2: YOLO11-seg training ──────────────────────────────────────────────
if [[ $FROM_STEP -le 2 ]]; then
    echo ""
    echo "[Step 2] Starting YOLO11-seg training..."
    python3 scripts/yolo_seg/train.py \
        --data "$WDS_ROOT/data.yaml" \
        --model yolo11m-seg.pt \
        --epochs 50 \
        --batch 32 \
        --device 0 \
        --workers 4 \
        --project "$WDS_ROOT/runs" \
        --name yolo11m-seg \
        2>&1 | tee "$LOG_DIR/train.log"
    echo "[Step 2] DONE."
fi

# ── Step 3: Predict + visualize on test sequences ────────────────────────────
if [[ $FROM_STEP -le 3 ]]; then
    echo ""
    echo "[Step 3] Running inference on test sequences..."
    BEST_PT="$WDS_ROOT/runs/yolo11m-seg/weights/best.pt"
    if [[ ! -f "$BEST_PT" ]]; then
        echo "  WARNING: $BEST_PT not found, trying last.pt"
        BEST_PT="$WDS_ROOT/runs/yolo11m-seg/weights/last.pt"
    fi
    python3 scripts/yolo_seg/predict_visualize.py \
        --model "$BEST_PT" \
        --views 0 7 14 21 28 35 \
        --step 60 \
        --conf 0.25 \
        2>&1 | tee "$LOG_DIR/predict.log"
    echo "[Step 3] DONE."
fi

# ── Step 4: Prepare matching data ────────────────────────────────────────────
if [[ $FROM_STEP -le 4 ]]; then
    echo ""
    echo "[Step 4] Preparing cross-view matching dataset..."
    python3 scripts/yolo_seg/prepare_matching_data.py \
        --step 60 \
        --num_views 6 \
        --augment 3 \
        --workers 8 \
        2>&1 | tee "$LOG_DIR/prepare_matching.log"
    echo "[Step 4] DONE."
fi

# ── Step 5: Train matching network ───────────────────────────────────────────
if [[ $FROM_STEP -le 5 ]]; then
    echo ""
    echo "[Step 5] Training cross-view matching network..."
    python3 scripts/yolo_seg/train_matching.py \
        --epochs 30 \
        --batch 32 \
        --lr 1e-4 \
        --device 0 \
        2>&1 | tee "$LOG_DIR/train_matching.log"
    echo "[Step 5] DONE."
fi

echo ""
echo "=== Pipeline complete ==="
