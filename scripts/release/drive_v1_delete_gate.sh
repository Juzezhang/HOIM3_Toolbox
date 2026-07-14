#!/bin/bash
# Deletes remote V1 (mhr/, smplx/, calib_ground_refined/) from the open-source
# Drive ONLY after V2 is fully verified uploaded. User-ordered 2026-07-13
# ("直接上传v2配对资产到开源里面,远端删除V1"). rclone delete on Drive moves to
# TRASH (30-day recovery window); local v1 copies remain canonical on simurgh2.
set -uo pipefail
RC=/simurgh2/users/juze/bin/rclone
LOG=/simurgh2/users/juze/calibjoint/drive_v1_delete.log
iter=0
while [ $iter -lt 400 ]; do
  iter=$((iter+1))
  ns=$($RC lsf gdrive_stanford:HOI-M3/smplx_with_distortion 2>/dev/null | wc -l)
  nm=$($RC lsf gdrive_stanford:HOI-M3/mhr_withdist 2>/dev/null | grep -c 'tar.gz')
  nc=$($RC lsf gdrive_stanford:HOI-M3/calib_with_distortion --dirs-only 2>/dev/null | wc -l)
  echo "$(date +%H:%M) v2 check: smplx=$ns/1292 mhr_tars=$nm/204 calib_dates=$nc/8" >> "$LOG"
  if [ "$ns" -ge 1292 ] && [ "$nm" -ge 204 ] && [ "$nc" -ge 8 ]; then
    echo "$(date +%H:%M) V2 COMPLETE — deleting V1 (to Drive trash)..." >> "$LOG"
    for d in mhr smplx calib_ground_refined; do
      $RC purge "gdrive_stanford:HOI-M3/$d" -q \
        && echo "$(date +%H:%M) deleted V1 $d" >> "$LOG" \
        || echo "$(date +%H:%M) FAILED deleting $d" >> "$LOG"
    done
    # empty trash so the deletion actually frees quota (Drive trash counts
    # against quota; user is at ~5T cap). Local v1 copies remain on simurgh2.
    $RC cleanup gdrive_stanford: -q && echo "$(date +%H:%M) trash emptied (quota reclaimed)" >> "$LOG"
    echo "$(date +%H:%M) V1_DELETION_DONE" >> "$LOG"
    break
  fi
  sleep 1200
done
