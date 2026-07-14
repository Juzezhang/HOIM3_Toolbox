#!/bin/bash
# Per-seq finish: aggregate tracked views -> npz -> LZ4 shards, NON-DESTRUCTIVELY
# (original mask_shards/<seq> is BACKED UP to mask_shards_pre_cutie/<seq> first).
# Usage: cutie_finish_seq.sh <seq>
# Only runs if all 42 views have .tracked_done. Safe to re-run (idempotent-ish).
set -euo pipefail
SEQ="${1:?seq required}"
D=/simurgh2/datasets/HOI-M3
ENV=/simurgh2/users/juze/anaconda3/envs/HOIM3_Toolbox/bin/python
CT=$D/cutie_tracking/$SEQ

# 1) require all 42 views tracked
missing=0
for v in $(seq 0 41); do [ -f "$CT/$v/.tracked_done" ] || missing=$((missing+1)); done
if [ $missing -gt 0 ]; then echo "[$SEQ] NOT READY: $missing/42 views missing .tracked_done"; exit 2; fi

# 2) aggregate -> mask_npz_cutie/<seq>/
echo "[$SEQ] aggregating..."
$ENV /simurgh/u/juze/code/HOIM3_Toolbox/scripts/regen/aggregate_cutie_to_npz.py --seq "$SEQ" --workers 8

# 3) BACKUP original shards (once), then convert
BAK=$D/mask_shards_pre_cutie/$SEQ
if [ ! -d "$BAK" ] && [ -d "$D/mask_shards/$SEQ" ]; then
  echo "[$SEQ] backing up original mask_shards -> $BAK"
  mkdir -p "$D/mask_shards_pre_cutie"
  cp -a "$D/mask_shards/$SEQ" "$BAK"
fi

TMP=$(mktemp -d /tmp/cutie_convert_${SEQ}_XXXX)
mkdir -p "$TMP/mask_npz"
ln -sfn "$D/mask_npz_cutie/$SEQ" "$TMP/mask_npz/$SEQ"
rm -rf "$D/mask_shards/$SEQ"
echo "[$SEQ] converting to LZ4 shards..."
$ENV /simurgh/u/juze/code/HOIM3_Toolbox/scripts/convert_masks_npz_to_lz4.py \
    --src_root "$TMP" --dst_root "$D" \
    --sequences "$SEQ" --compression_level 6 --num_workers 12

touch "$D/mask_shards/$SEQ/.merged_1080p_done"
touch "$D/mask_shards/$SEQ/.restored_swap24"
touch "$D/mask_shards/$SEQ/.cutie_retracked"
echo "[$SEQ] DONE. Original preserved at $BAK"
