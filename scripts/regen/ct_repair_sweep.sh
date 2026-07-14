#!/bin/bash
# ct_ (Cutie mask-tracking) repair sweep: for each of the 41 seqs, find views
# without .tracked_done AND with no queued/running ct_ task, resubmit just those
# view indices. Run in a loop (30 min) — mask-fix previously had NO repair path,
# so single-view failures stalled seqs at 41/42 forever.
set -uo pipefail
CT=/simurgh2/datasets/HOI-M3/cutie_tracking
ARR=/simurgh2/users/juze/calibjoint/cutie_track_seq_array.sh
LOG=/simurgh2/users/juze/calibjoint/ct_repair.log
SEQS=(livingroom_data03 livingroom_data04 livingroom_data30 livingroom_data31 livingroom_data32 livingroom_data48 livingroom_data07 livingroom_data11 livingroom_data21 livingroom_data33 livingroom_data34 bedroom_data05 bedroom_data35 bedroom_data01 bedroom_data03 bedroom_data15 bedroom_data30 bedroom_data31 bedroom_data32 bedroom_data33 bedroom_data34 diningroom_data09 diningroom_data11 diningroom_data01 diningroom_data06 office_data18 office_data61 office_data03 office_data25 office_data26 office_data27 office_data29 office_data30 office_data31 office_data37 office_data38 office_data39 office_data43 office_data44 office_data45 office_data50)
iter=0
while [ $iter -lt 300 ]; do
  iter=$((iter+1)); anyleft=0
  for s in "${SEQS[@]}"; do
    d=$(ls $CT/$s/*/.tracked_done 2>/dev/null | wc -l)
    [ "$d" -eq 42 ] && continue
    anyleft=1
    # queued/running view indices for this seq
    Q=$(squeue -u juze -h -o "%j %K" 2>/dev/null | awk -v s="ct_$s" '$1==s{print $2}' | tr ',' '\n')
    # skip if array still has bracketed pending ranges
    squeue -u juze -h -o "%i %j" 2>/dev/null | grep "ct_$s" | grep -q '\[' && continue
    missing=""
    for v in $(seq 0 41); do
      [ -f "$CT/$s/$v/.tracked_done" ] && continue
      echo "$Q" | grep -qx "$v" && continue
      missing="$missing,$v"
    done
    missing=${missing#,}
    if [ -n "$missing" ]; then
      jid=$(SEQ=$s sbatch --parsable --export=ALL,SEQ=$s --job-name=ct_$s --array=$missing "$ARR" 2>/dev/null) \
        && echo "$(date +%H:%M) [ct-repair] $s views [$missing] -> $jid" | tee -a "$LOG"
    fi
  done
  [ $anyleft -eq 0 ] && { echo "$(date +%H:%M) ALL 41 SEQS FULLY TRACKED" | tee -a "$LOG"; break; }
  sleep 1800
done
