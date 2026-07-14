#!/bin/bash
# Incremental GCS uploader (user-approved 2026-07-13 "合并好了的话可以一边上传着").
# Every 30 min:
#  1. rsync smplx_with_distortion/  (diffs only -> each newly committed seq flows up)
#  2. one-time rsync smplx_with_distortion_mhr_bak/ (MHR originals of replaced seqs)
#  3. per seq with mask_shards/.cutie_retracked: server-side backup of the REMOTE
#     original (mask_shards -> mask_shards_pre_cutie, no egress), then rsync the
#     rebuilt local shards up. Stamped in calibjoint/gcs_stamps/.
set -uo pipefail
export GSUTIL=/simurgh2/users/juze/anaconda3/envs/SAMPA/bin/gsutil
export BOTO_CONFIG=/simurgh/u/juze/.boto_gcs
B=gs://data-storage-0/HOI-M3
D=/simurgh2/datasets/HOI-M3
ST=/simurgh2/users/juze/calibjoint/gcs_stamps
LOG=/simurgh2/users/juze/calibjoint/gcs_upload.log
mkdir -p "$ST"
iter=0
while [ $iter -lt 200 ]; do
  iter=$((iter+1))
  # 1. smplx_with_distortion (always; rsync only moves diffs)
  $GSUTIL -m -q rsync -r "$D/smplx_with_distortion" "$B/smplx_with_distortion" \
    && echo "$(date +%H:%M) smplx_with_distortion synced" >> "$LOG" \
    || echo "$(date +%H:%M) smplx_with_distortion sync FAILED" >> "$LOG"
  # 2. backup dir (once)
  if [ ! -f "$ST/mhr_bak.done" ]; then
    $GSUTIL -m -q rsync -r "$D/smplx_with_distortion_mhr_bak" "$B/smplx_with_distortion_mhr_bak" \
      && touch "$ST/mhr_bak.done" && echo "$(date +%H:%M) mhr_bak synced" >> "$LOG"
  fi
  # 3. rebuilt shards, per seq
  for f in "$D"/mask_shards/*/.cutie_retracked; do
    [ -e "$f" ] || continue
    seq=$(basename "$(dirname "$f")")
    [ -f "$ST/shards_$seq.done" ] && continue
    # server-side backup of remote original (skip if backup already there)
    if ! $GSUTIL -q stat "$B/mask_shards_pre_cutie/$seq/meta.json" 2>/dev/null; then
      $GSUTIL -m -q cp -r "$B/mask_shards/$seq" "$B/mask_shards_pre_cutie/" 2>>"$LOG" \
        && echo "$(date +%H:%M) [shards] $seq remote original backed up (server-side)" >> "$LOG"
    fi
    $GSUTIL -m -q rsync -r -d "$D/mask_shards/$seq" "$B/mask_shards/$seq" \
      && touch "$ST/shards_$seq.done" \
      && echo "$(date +%H:%M) [shards] $seq uploaded (rebuilt, with objects)" >> "$LOG" \
      || echo "$(date +%H:%M) [shards] $seq upload FAILED" >> "$LOG"
  done
  sleep 1800
done
