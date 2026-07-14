#!/bin/bash
# Drive open-source v2 uploader: mhr_withdist tar.gz as they are produced +
# smplx_with_distortion re-sync every cycle (new DenseFit commits flow up).
set -uo pipefail
RC=/simurgh2/users/juze/bin/rclone
D=/simurgh2/datasets/HOI-M3
TG=/simurgh2/users/juze/mhr_withdist_targz
LOG=/simurgh2/users/juze/calibjoint/drive_v2_upload.log
iter=0
while [ $iter -lt 200 ]; do
  iter=$((iter+1))
  # tars (copy is incremental: size+mtime)
  n=$(ls $TG/*.tar.gz 2>/dev/null | wc -l)
  if [ "$n" -gt 0 ]; then
    $RC copy "$TG" gdrive_stanford:HOI-M3/mhr_withdist --include "*.tar.gz" --transfers 6 -q \
      && echo "$(date +%H:%M) mhr_withdist tars synced (local $n)" >> "$LOG"
  fi
  # smplx re-sync
  $RC copy "$D/smplx_with_distortion" gdrive_stanford:HOI-M3/smplx_with_distortion --transfers 6 -q \
    && echo "$(date +%H:%M) smplx re-synced" >> "$LOG"
  # done condition: all 204 tars local AND uploaded count matches
  if [ "$n" -eq 204 ]; then
    ru=$($RC lsf gdrive_stanford:HOI-M3/mhr_withdist 2>/dev/null | grep -c 'tar.gz')
    [ "$ru" -eq 204 ] && { echo "$(date +%H:%M) ALL_V2_MHR_UPLOADED" >> "$LOG"; break; }
  fi
  sleep 1200
done
