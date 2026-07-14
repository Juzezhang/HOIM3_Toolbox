#!/bin/bash
# Per-seq DenseFit fan-out: for each of the 41 swap-prone seqs, once its Cutie
# tracking is complete (42 views .tracked_done), prep + submit its DenseFit window
# array on sc-freegpu (uncapped -> fills all idle cards). Records array jids.
# Concat+commit is done SEPARATELY (densefit_concat.py --commit) once a seq's
# windows all finish, to keep the merge gated/inspectable.
set -uo pipefail
ENV=/simurgh2/users/juze/anaconda3/envs/densefit/bin/python
CT=/simurgh2/datasets/HOI-M3/cutie_tracking
ARR=/simurgh2/users/juze/calibjoint/densefit_window_array.sh
JIDLOG=/simurgh2/users/juze/calibjoint/densefit_fanout_jids.txt
W=300
SEQS=(livingroom_data03 livingroom_data04 livingroom_data30 livingroom_data31 livingroom_data32 livingroom_data48 livingroom_data07 livingroom_data11 livingroom_data21 livingroom_data33 livingroom_data34 bedroom_data05 bedroom_data35 bedroom_data01 bedroom_data03 bedroom_data15 bedroom_data30 bedroom_data31 bedroom_data32 bedroom_data33 bedroom_data34 diningroom_data09 diningroom_data11 diningroom_data01 diningroom_data06 office_data18 office_data61 office_data03 office_data25 office_data26 office_data27 office_data29 office_data30 office_data31 office_data37 office_data38 office_data39 office_data43 office_data44 office_data45 office_data50)
iter=0
while [ $iter -lt 2000 ]; do
  iter=$((iter+1)); pending=0
  for s in "${SEQS[@]}"; do
    grep -qE "^$s [0-9]" "$JIDLOG" 2>/dev/null && continue   # already submitted DenseFit
    # need mask_shards meta (some lack it) AND all 42 cutie views tracked
    d=0; for v in $(seq 0 41); do [ -f "$CT/$s/$v/.tracked_done" ] && d=$((d+1)); done
    if [ $d -lt 42 ]; then pending=1; continue; fi
    # prep + submit
    $ENV /simurgh2/users/juze/calibjoint/densefit_prep_seq.py "$s" >/dev/null 2>&1 || { echo "PREPFAIL $s" >> "$JIDLOG"; continue; }
    NF=$($ENV -c "import json;print(json.load(open('/simurgh2/users/juze/calibjoint/prepinfo_$s.json'))['frames'])")
    NWIN=$(( (NF + W - 1) / W ))
    jid=$(SEQ=$s WIN=$W sbatch --parsable --export=ALL,SEQ=$s,WIN=$W --job-name=mm_$s --array=0-$((NWIN-1)) "$ARR" 2>/dev/null)
    echo "$s $jid nwin=$NWIN" >> "$JIDLOG"
    echo "$(date +%H:%M) submitted DenseFit $s -> $jid ($NWIN windows)"
  done
  [ $pending -eq 0 ] && { echo "ALL_SUBMITTED $(date)"; break; }
  sleep 120
done
