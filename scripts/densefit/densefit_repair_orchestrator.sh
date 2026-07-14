#!/bin/bash
# Repair loop: every 30 min, for each seq with a submitted DenseFit array, find
# windows lacking ma_3d output AND having no queued/running task, resubmit just
# those indices (runner exits fast on already-done windows). Handles the two
# observed failure modes: flaky no-GPU nodes (exit 66) and pre-fix OOM windows.
set -uo pipefail
ENV=/simurgh2/users/juze/anaconda3/envs/densefit/bin/python
MM=${DENSEFIT_ROOT:-/path/to/densefit_workspace}
ARR=/simurgh2/users/juze/calibjoint/densefit_window_array.sh
JIDLOG=/simurgh2/users/juze/calibjoint/densefit_fanout_jids.txt
LOG=/simurgh2/users/juze/calibjoint/densefit_repair.log
W=300
iter=0
while [ $iter -lt 200 ]; do
  iter=$((iter+1))
  while read -r seq jid rest; do
    case "$seq" in livingroom_*|bedroom_*|diningroom_*|office_*) ;; *) continue;; esac
    NF=$($ENV -c "import json;print(json.load(open('/simurgh2/users/juze/calibjoint/prepinfo_$seq.json'))['frames'])" 2>/dev/null) || continue
    NWIN=$(( (NF + W - 1) / W ))
    # windows still queued/running for this seq (any array)
    QUEUED=$(squeue -u juze -h -o "%j %K" 2>/dev/null | awk -v s="mm_$seq" -v r="mmR_$seq" '$1==s||$1==r{print $2}' | tr ',' '\n')
    missing=""
    for w in $(seq 0 $((NWIN-1))); do
      t=$(printf '%s_full_w%04d' "$seq" "$w")
      [ -n "$(find $MM/output/ma_3d/$t -name 'smplx_params_body_id-00.npz' 2>/dev/null | head -1)" ] && continue
      echo "$QUEUED" | grep -qx "$w" && continue
      # also treat [x-y] pending ranges conservatively: if any pending range exists, skip repair this round
      missing="$missing,$w"
    done
    # skip if the array still has pending bracketed tasks (e.g. 16140044_[0-72])
    if squeue -u juze -h -o "%i %j" 2>/dev/null | grep "mm_$seq" | grep -q '\['; then continue; fi
    missing=${missing#,}
    if [ -n "$missing" ]; then
      nmiss=$(echo "$missing" | tr ',' '\n' | wc -l)
      # small repair sets go to the simurgh fast lane (sc-freegpu FIFO starves them)
      RARR="$ARR"; TAG=mm_$seq
      if [ "$nmiss" -le 8 ]; then RARR=/simurgh2/users/juze/calibjoint/densefit_window_array_sim.sh; TAG=mmR_$seq; fi
      njid=$(SEQ=$seq WIN=$W sbatch --parsable --export=ALL,SEQ=$seq,WIN=$W --job-name=$TAG --array=$missing "$RARR" 2>/dev/null) \
        && echo "$(date +%H:%M) [repair] $seq resubmitted [$missing] via $(basename $RARR) -> $njid" | tee -a "$LOG"
    fi
  done < "$JIDLOG"
  sleep 1800
done
