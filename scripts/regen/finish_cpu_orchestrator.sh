#!/bin/bash
# CPU-queue finish orchestrator: when a seq has 42/42 .tracked_done and is not
# yet .cutie_retracked and has no queued finish job -> sbatch a per-seq finish
# job on sc-freecpu (aggregate -> backup -> LZ4). Replaces the login-node
# serial finisher (was ~4-5h/seq SERIAL; now all seqs run in parallel).
set -uo pipefail
D=/simurgh2/datasets/HOI-M3
CT=$D/cutie_tracking
LOG=/simurgh2/users/juze/calibjoint/finish_cpu_orch.log
SEQS=(livingroom_data03 livingroom_data04 livingroom_data30 livingroom_data31 livingroom_data32 livingroom_data48 livingroom_data07 livingroom_data11 livingroom_data21 livingroom_data33 livingroom_data34 bedroom_data05 bedroom_data35 bedroom_data01 bedroom_data03 bedroom_data15 bedroom_data30 bedroom_data31 bedroom_data32 bedroom_data33 bedroom_data34 diningroom_data09 diningroom_data11 diningroom_data01 diningroom_data06 office_data18 office_data61 office_data03 office_data25 office_data26 office_data27 office_data29 office_data30 office_data31 office_data37 office_data38 office_data39 office_data43 office_data44 office_data45 office_data50)
iter=0
while [ $iter -lt 600 ]; do
  iter=$((iter+1)); pending=0
  for s in "${SEQS[@]}"; do
    [ -f "$D/mask_shards/$s/.cutie_retracked" ] && continue
    pending=1
    d=$(ls $CT/$s/*/.tracked_done 2>/dev/null | wc -l)
    [ "$d" -eq 42 ] || continue
    squeue -u juze -h -o "%j" 2>/dev/null | grep -qx "fin_$s" && continue
    jid=$(SEQ=$s sbatch --parsable --export=ALL,SEQ=$s --job-name=fin_$s \
          /simurgh2/users/juze/calibjoint/run_finish_cpu.sh 2>/dev/null) \
      && echo "$(date +%H:%M) [fin] $s -> sc-freecpu job $jid" | tee -a "$LOG"
  done
  [ $pending -eq 0 ] && { echo "$(date +%H:%M) ALL SHARDS REBUILT" | tee -a "$LOG"; break; }
  sleep 900
done
