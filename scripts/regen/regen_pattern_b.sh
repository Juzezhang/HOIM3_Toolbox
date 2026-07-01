#!/bin/bash
# Regenerate masks for 12 "Pattern B" HOI-M3 sequences with missing tail views.
#
# For each seq, sequentially:
#   1. Run inference_masks.py to (re)generate mask_npz_generated/<seq>/*.npz
#   2. Run fix_one_seq_v2.sh <seq> to convert+upgrade+sentinel
#
# Order: smallest missing-view count first, biggest last.
# Logs: /scr/juze/regen_pattern_b.log
#
# Usage:
#   /scr/juze/regen_pattern_b.sh           # run all 12 in order
#   /scr/juze/regen_pattern_b.sh <seq>...  # run only specific seqs

set -u

PY_TOOLBOX=/simurgh2/users/juze/anaconda3/envs/HOIM3_Toolbox/bin/python
INFER=/simurgh/u/juze/code/HOIM3_Toolbox/scripts/yolo_seg/inference_masks.py
FIX=/simurgh/u/juze/code/HOIM3_Toolbox/scripts/regen/fix_one_seq_v2.sh
GPU=${GPU:-0}
LOG=/scr/juze/regen_pattern_b.log
SENTINEL_DIR=/scr/juze/regen_pattern_b_done
mkdir -p "$SENTINEL_DIR"

# Order: smallest workload first
DEFAULT_SEQS=(
    office_data45        # 7 missing
    diningroom_data09    # 11
    bedroom_data15       # 13
    bedroom_data30       # 18
    bedroom_data32       # 19
    bedroom_data31       # 20
    bedroom_data33       # 22
    bedroom_data34       # 22
    bedroom_data05       # 24
    bedroom_data35       # 25
    livingroom_data48    # 28
    diningroom_data11    # 29
)

if [ $# -gt 0 ]; then
    SEQS=("$@")
else
    SEQS=("${DEFAULT_SEQS[@]}")
fi

echo "============================================================" | tee -a "$LOG"
echo "$(date) [regen_pattern_b] START gpu=$GPU seqs=${SEQS[*]}" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

for SEQ in "${SEQS[@]}"; do
    SENTINEL="$SENTINEL_DIR/$SEQ.done"
    if [ -f "$SENTINEL" ]; then
        echo "$(date) [$SEQ] already done (sentinel exists), skipping" | tee -a "$LOG"
        continue
    fi

    t_start=$(date +%s)
    echo "------------------------------------------------------------" | tee -a "$LOG"
    echo "$(date) [$SEQ] STAGE 1: YOLO+ReID inference (gpu=$GPU)" | tee -a "$LOG"

    CUDA_VISIBLE_DEVICES=$GPU $PY_TOOLBOX "$INFER" \
        --gpu 0 --sequences "$SEQ" \
        >> "$LOG" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "$(date) [$SEQ] inference FAILED rc=$rc — skipping fix stage" | tee -a "$LOG"
        continue
    fi
    t_inf=$(($(date +%s) - t_start))
    echo "$(date) [$SEQ] inference done in ${t_inf}s" | tee -a "$LOG"

    n_npz=$(ls /simurgh2/datasets/HOI-M3/mask_npz_generated/$SEQ/ 2>/dev/null | wc -l)
    echo "$(date) [$SEQ] mask_npz_generated has $n_npz npz files" | tee -a "$LOG"

    echo "$(date) [$SEQ] STAGE 2: convert+upgrade via fix_one_seq_v2.sh" | tee -a "$LOG"
    t_fix0=$(date +%s)
    bash "$FIX" "$SEQ" >> "$LOG" 2>&1
    rc=$?
    t_fix=$(($(date +%s) - t_fix0))
    if [ $rc -ne 0 ]; then
        echo "$(date) [$SEQ] fix_one_seq_v2 FAILED rc=$rc" | tee -a "$LOG"
        continue
    fi
    echo "$(date) [$SEQ] fix done in ${t_fix}s" | tee -a "$LOG"

    # Cleanup tmp dirs (defensive — fix_one_seq_v2 already cleans)
    rm -rf /scr/juze/${SEQ}_fix_src /scr/juze/${SEQ}_fix_dst 2>/dev/null

    t_total=$(($(date +%s) - t_start))
    echo "$(date) [$SEQ] DONE total=${t_total}s (inf=${t_inf}s fix=${t_fix}s)" | tee -a "$LOG"
    touch "$SENTINEL"
done

echo "============================================================" | tee -a "$LOG"
echo "$(date) [regen_pattern_b] ALL DONE" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
