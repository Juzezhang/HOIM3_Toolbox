#!/bin/bash
# Local simurgh5 GPU worker for Cutie tracking.
# Polls a shared pending file, claims (seq, view) via mkdir lock, runs Cutie.
#
# Usage:  cutie_track_local_worker.sh <GPU_ID>   (also reads env: PENDING_FILE)
#
# Atomic claim: try mkdir /scr/juze/cutie_track_locks/<seq>_<view>.lock
# Remove the line from pending only on successful claim.

set -u
GPU_ID="${1:?gpu_id required}"
PENDING_FILE="${PENDING_FILE:-/scr/juze/cutie_tracking_pending.txt}"
DONE_FILE="${DONE_FILE:-/scr/juze/cutie_tracking_done.list}"
LOCK_DIR="${LOCK_DIR:-/scr/juze/cutie_track_locks}"
LOG_DIR="${LOG_DIR:-/scr/juze/swap24_cleanup_logs}"
OUTPUT_ROOT_SENTINEL="${OUTPUT_ROOT_SENTINEL:-/simurgh2/datasets/HOI-M3/cutie_tracking}"

mkdir -p "$LOCK_DIR" "$LOG_DIR"
touch "$DONE_FILE"

CONDA_BASE=/simurgh2/users/juze/anaconda3
set +u
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate HHOI-Toolkit
set -u

export CUDA_VISIBLE_DEVICES="$GPU_ID"

WORKER_LOG="$LOG_DIR/cutie_local_gpu${GPU_ID}.log"
exec >> "$WORKER_LOG" 2>&1

echo "$(date) [cutie_local gpu=$GPU_ID] START pid=$$ pending=$PENDING_FILE"

cd /simurgh/u/juze/code/HOIM3_Toolbox

while [ -s "$PENDING_FILE" ]; do
    line=$(head -1 "$PENDING_FILE" 2>/dev/null || true)
    if [ -z "$line" ]; then sleep 2; continue; fi
    seq=$(echo "$line" | awk '{print $1}')
    view=$(echo "$line" | awk '{print $2}')
    if [ -z "$seq" ] || [ -z "$view" ]; then
        sed -i '1d' "$PENDING_FILE" 2>/dev/null || true
        continue
    fi
    lock="$LOCK_DIR/${seq}_${view}.lock"
    sentinel="$OUTPUT_ROOT_SENTINEL/$seq/$view/.tracked_done"
    if [ -f "$sentinel" ]; then
        grep -vxF "$seq $view" "$PENDING_FILE" > "${PENDING_FILE}.tmp$$" 2>/dev/null || true
        mv "${PENDING_FILE}.tmp$$" "$PENDING_FILE" 2>/dev/null || true
        continue
    fi
    if mkdir "$lock" 2>/dev/null; then
        # Remove the line we claimed (exact line match — avoid prefix collisions)
        grep -vxF "$seq $view" "$PENDING_FILE" > "${PENDING_FILE}.tmp$$" 2>/dev/null || true
        mv "${PENDING_FILE}.tmp$$" "$PENDING_FILE" 2>/dev/null || true
        echo "$(date) [cutie_local gpu=$GPU_ID] CLAIM $seq v$view"
        if python scripts/regen/cutie_track_one_view.py \
            --seq "$seq" --view "$view" \
            --ref_root /simurgh2/datasets/HOI-M3/cutie_refs \
            --video_root /simurgh2/datasets/HOI-M3/videos \
            --output_root /simurgh2/datasets/HOI-M3/cutie_tracking \
            --max_internal_size 480 \
            --frame_stride 3 \
            --device "cuda:0"; then
            echo "$seq $view OK $(date +%s)" >> "$DONE_FILE"
            echo "$(date) [cutie_local gpu=$GPU_ID] DONE $seq v$view"
        else
            rc=$?
            echo "$seq $view FAIL_rc${rc} $(date +%s)" >> "$DONE_FILE"
            echo "$(date) [cutie_local gpu=$GPU_ID] FAIL $seq v$view rc=$rc"
        fi
        rmdir "$lock" 2>/dev/null || true
    else
        # Lock exists (another worker / sbatch claimed it). Remove this line from pending so we don't busy-loop.
        grep -vxF "$seq $view" "$PENDING_FILE" > "${PENDING_FILE}.tmp$$" 2>/dev/null || true
        mv "${PENDING_FILE}.tmp$$" "$PENDING_FILE" 2>/dev/null || true
    fi
done

echo "$(date) [cutie_local gpu=$GPU_ID] EXIT — pending empty"
