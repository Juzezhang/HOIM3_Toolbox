#!/bin/bash
# Convert from mask_npz_generated → 720p named-key shard, then upgrade to 1080p, write sentinel.
set -eu
SEQ=${1:?Usage: $0 <seq_name>}
LOG=/scr/juze/fix_${SEQ}.log
PY_TOOLBOX=/simurgh2/users/juze/anaconda3/envs/HOIM3_Toolbox/bin/python
LOCK_DIR=/scr/juze/merge_locks
LOCK_FILE="$LOCK_DIR/$SEQ.lock"
mkdir -p "$LOCK_DIR"
echo "$(date) [$SEQ] FIX START" | tee -a "$LOG"
attempts=0
while ! mkdir "$LOCK_FILE" 2>/dev/null; do
    attempts=$((attempts+1))
    [ "$attempts" -gt 600 ] && { echo "$(date) [$SEQ] lock timeout" | tee -a "$LOG"; exit 1; }
    sleep 6
done
trap "rmdir $LOCK_FILE 2>/dev/null" EXIT
GEN_DIR=/simurgh2/datasets/HOI-M3/mask_npz_generated/$SEQ
[ ! -d "$GEN_DIR" ] && { echo "$(date) [$SEQ] no gen dir" | tee -a "$LOG"; exit 1; }
BAD=/simurgh2/datasets/HOI-M3/mask_shards/$SEQ
TMP_SRC=/scr/juze/${SEQ}_fix_src
TMP_DST=/scr/juze/${SEQ}_fix_dst
rm -rf "$TMP_SRC" "$TMP_DST"
mkdir -p "$TMP_SRC/mask_npz" "$TMP_DST"
ln -s "$GEN_DIR" "$TMP_SRC/mask_npz/$SEQ"
echo "$(date) [$SEQ] running converter (mask_npz_generated → 720p named shard)..." | tee -a "$LOG"
$PY_TOOLBOX /simurgh/u/juze/code/HOIM3_Toolbox/scripts/convert_masks_npz_to_lz4.py \
    --src_root "$TMP_SRC" --dst_root "$TMP_DST" --sequences "$SEQ" \
    --num_workers 4 --compression_level 6 >> "$LOG" 2>&1
NEW="$TMP_DST/mask_shards/$SEQ"
[ ! -f "$NEW/meta.json" ] && { echo "$(date) [$SEQ] convert FAILED" | tee -a "$LOG"; exit 1; }
echo "$(date) [$SEQ] convert OK, replacing BAD shard..." | tee -a "$LOG"
rm -rf "$BAD"
mv "$NEW" "$BAD"
rm -f "$BAD/.merged_1080p_done"
echo "$(date) [$SEQ] running upgrade 720p → 1080p..." | tee -a "$LOG"
$PY_TOOLBOX /simurgh/u/juze/code/HOIM3_Toolbox/scripts/regen/upgrade_to_1080p.py --seqs "$SEQ" >> "$LOG" 2>&1
[ ! -f "$BAD/.merged_1080p_done" ] && { echo "$(date) [$SEQ] upgrade FAILED (no sentinel)" | tee -a "$LOG"; exit 1; }
# Clear stale mono outputs so v17/v19/v21 followers re-run
for v in 17 19 21; do rm -rf "/scr/juze/datasets/HOI-M3/mhr_mono/$SEQ/$v" 2>/dev/null; done
rm -f /scr/juze/datasets/HOI-M3/mono_view_17_19_done/$SEQ.done /scr/juze/datasets/HOI-M3/mono_view_21_done/$SEQ.done 2>/dev/null
rm -rf "$TMP_SRC" "$TMP_DST"
echo "$(date) [$SEQ] FIX DONE" | tee -a "$LOG"
