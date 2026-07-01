#!/bin/bash
# Post-Cutie pipeline for bedroom_data01 fix:
#   1) Wait for all 20 BAD view .tracked_done sentinels
#   2) Aggregate Cutie outputs + existing shard → staging mask_npz
#   3) Convert staging mask_npz → temp mask_shards_bedroom_data01_fixed
#   4) Run viz on the temp shards
#
# Does NOT touch /simurgh2/datasets/HOI-M3/mask_shards/bedroom_data01/.
#
# Run in background:
#   nohup bash post_bedroom_data01.sh </dev/null \
#     >/scr/juze/swap24_cleanup_logs/post_bedroom_data01.log 2>&1 &

set -u

SEQ=bedroom_data01
BAD_VIEWS=(0 1 2 5 6 7 8 9 23 27 28 29 30 32 33 35 37 38 40 41)
CUTIE_OUT=/scr/juze/datasets/HOI-M3/cutie_tracking_bedroom_data01
STAGING_NPZ=/simurgh2/datasets/HOI-M3/mask_npz_cutie/bedroom_data01_fixed
TMP_SRC_ROOT=/scr/juze/bedroom_data01_fixed_src
TMP_DST_ROOT=/simurgh2/datasets/HOI-M3/mask_shards_bedroom_data01_fixed_dstparent
SHARD_OUT=/simurgh2/datasets/HOI-M3/mask_shards_bedroom_data01_fixed
LOG=/scr/juze/swap24_cleanup_logs/post_bedroom_data01.log
CONDA_BASE=/simurgh2/users/juze/anaconda3

mkdir -p "$(dirname $LOG)" "$STAGING_NPZ" "$TMP_SRC_ROOT/mask_npz" "$TMP_DST_ROOT"

set +u
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate HOIM3_Toolbox
set -u

log() { echo "$(date '+%F %T') [bedroom_post] $*"; }

# 1) Wait for all 20 sentinels
log "waiting for ${#BAD_VIEWS[@]} bedroom_data01 view sentinels..."
while true; do
    missing=()
    for v in "${BAD_VIEWS[@]}"; do
        if [ ! -f "$CUTIE_OUT/$SEQ/$v/.tracked_done" ]; then
            missing+=("$v")
        fi
    done
    if [ ${#missing[@]} -eq 0 ]; then
        log "all ${#BAD_VIEWS[@]} sentinels present"
        break
    fi
    log "waiting on views: ${missing[*]}"
    sleep 120
done

# 2) Aggregate
log "aggregating Cutie outputs + shard → $STAGING_NPZ"
python /simurgh/u/juze/code/HOIM3_Toolbox/scripts/regen/aggregate_cutie_bedroom_to_npz.py \
    --workers 12
rc=$?
if [ $rc -ne 0 ]; then
    log "aggregate FAIL rc=$rc"
    exit 1
fi

# 3) Convert to LZ4 shard
# The converter writes to <dst_root>/mask_shards/<seq>. We want
# /simurgh2/datasets/HOI-M3/mask_shards_bedroom_data01_fixed/, so place
# dst_root = /simurgh2/datasets/HOI-M3 with a sentinel sequence name.
# Instead: use a TMP_DST_ROOT and then rename the output dir.
mkdir -p "$TMP_SRC_ROOT/mask_npz"
ln -sfn "$STAGING_NPZ" "$TMP_SRC_ROOT/mask_npz/$SEQ"
rm -rf "$TMP_DST_ROOT/mask_shards/$SEQ"
mkdir -p "$TMP_DST_ROOT/mask_shards"

log "converting → $TMP_DST_ROOT/mask_shards/$SEQ"
python /simurgh/u/juze/code/HOIM3_Toolbox/scripts/convert_masks_npz_to_lz4.py \
    --src_root "$TMP_SRC_ROOT" \
    --dst_root "$TMP_DST_ROOT" \
    --sequences "$SEQ" \
    --compression_level 6 \
    --num_workers 12
rc=$?
if [ $rc -ne 0 ]; then
    log "convert FAIL rc=$rc"
    exit 1
fi

# Move shards to final temp location (separate dir, NOT overwriting real shards)
rm -rf "$SHARD_OUT"
mkdir -p "$(dirname $SHARD_OUT)"
mv "$TMP_DST_ROOT/mask_shards/$SEQ" "$SHARD_OUT"
log "shards staged at $SHARD_OUT"

# 4) Run viz pointing at the temp shard dir.
# viz_restored_mask.py uses hardcoded SHARD_ROOT = /simurgh2/datasets/HOI-M3/mask_shards/<seq>.
# We make a tmp copy with SHARD_ROOT overridden and run that.
VIZ_SCRIPT=/scr/juze/swap24_cleanup_logs/viz_restored_mask.py
VIZ_BEDROOM=/scr/juze/swap24_cleanup_logs/viz_restored_mask_bedroom_fixed.py
sed "s|^SHARD_ROOT = f\"/simurgh2/datasets/HOI-M3/mask_shards/{SEQ}\"|SHARD_ROOT = f\"$SHARD_OUT\"|" \
    "$VIZ_SCRIPT" > "$VIZ_BEDROOM"
chmod +x "$VIZ_BEDROOM"

log "running viz → /scr/juze/swap24_cleanup_logs/viz_bedroom_data01_fixed.mp4"
python "$VIZ_BEDROOM" "$SEQ" 30 \
    /scr/juze/swap24_cleanup_logs/viz_bedroom_data01_fixed.mp4
log "DONE — viz at /scr/juze/swap24_cleanup_logs/viz_bedroom_data01_fixed.mp4"
