#!/bin/bash
# Self-managing merge campaign monitor. Every 10 min:
#  - run salvage_mtmp.sh (backstop for any crash-at-swap; staleness-guarded, race-safe)
#  - keep total /simurgh2 NFS readers ~TARGET by bumping merge throttle as giants drain
#  - track 37-pack progress + merge done/fail; alert if a merge fails WITHOUT being salvaged
# Exits (to notify the parent) on: +PROG_STEP packs, an un-salvaged failure, or TIMEOUT.
set -u
PACK=/simurgh2/datasets/HOI-M3/mhr_mono_packed
PY=/simurgh2/users/juze/anaconda3/envs/mvbodyfit/bin/python
SAL=/simurgh/u/juze/code/HOIM3_Toolbox/scripts/regen/salvage_mtmp.sh
LOG=/simurgh/u/juze/regen_logs/merge_campaign.log
TARGET=10; PROG_STEP=15; MAXIT=24   # 24*10min = 4h then report
MJOB=$(squeue -u juze --name=merge37 -h -o '%A' 2>/dev/null|sort -u|head -1)
c37(){ $PY -c "import glob,numpy as np
n=0
for m in glob.glob('$PACK/*/person0/meta.npz'):
  try:
    if int(np.load(m)['views'].shape[0])==37: n+=1
  except: pass
print(n)"; }
base=$(c37); echo "$(date) [campaign] START base37=$base merge_job=$MJOB" >> "$LOG"
for i in $(seq 1 $MAXIT); do
  sleep 600
  bash "$SAL" >> "$LOG" 2>&1
  giants=$(squeue -u juze --name=pack37tar,pack37nm -h -t R 2>/dev/null|wc -l)
  mR=$(squeue -u juze --name=merge37 -h -t R 2>/dev/null|wc -l)
  now=$(c37)
  # adjust merge throttle: target total readers = TARGET
  want=$((TARGET - giants)); [ $want -lt 4 ] && want=4; [ $want -gt 10 ] && want=10
  cur=$(scontrol show job "$MJOB" 2>/dev/null | grep -oE 'ArrayTaskThrottle=[0-9]+' | grep -oE '[0-9]+' | head -1)
  if [ -n "$cur" ] && [ "$want" != "$cur" ]; then scontrol update arraytaskthrottle=$want jobid=$MJOB 2>/dev/null && echo "$(date) [campaign] merge throttle $cur->$want (giants=$giants)" >> "$LOG"; fi
  # detect un-salvaged failures: rc=1 lines whose seq is still NOT 37-view and has no .mtmp
  fails=$(grep -hE "merge37 rc=1" /simurgh/u/juze/regen_logs/merge37_*.out 2>/dev/null | grep -oE "[a-z]+_data[0-9]+" | sort -u)
  badfail=""
  for s in $fails; do
    v=$($PY -c "import numpy as np;print(int(np.load('$PACK/$s/person0/meta.npz')['views'].shape[0]))" 2>/dev/null||echo 0)
    [ "$v" != "37" ] && [ ! -d "$PACK/$s.mtmp" ] && badfail="$badfail $s"
  done
  echo "$(date) [campaign min$((i*10))] 37pack=$now (base=$base) giants=$giants mergeR=$mR throttle=$want badfail=[$badfail]" >> "$LOG"
  if [ -n "$badfail" ]; then echo "ALERT: un-salvaged merge failures:$badfail" >> "$LOG"; echo "CAMPAIGN_EXIT reason=UNSALVAGED_FAIL badfail=$badfail 37pack=$now"; exit 0; fi
  if [ $((now-base)) -ge $PROG_STEP ]; then echo "CAMPAIGN_EXIT reason=PROGRESS 37pack=$now (+$((now-base))) giants=$giants mergeR=$mR"; exit 0; fi
done
echo "CAMPAIGN_EXIT reason=TIMEOUT 37pack=$(c37) (base=$base)"
