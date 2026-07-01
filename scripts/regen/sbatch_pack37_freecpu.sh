#!/bin/bash
#SBATCH --job-name=pack37fc
#SBATCH --partition=sc-freecpu
#SBATCH --account=simurgh
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/simurgh/u/juze/regen_logs/pack37fc_%A_%a.out
#SBATCH --error=/simurgh/u/juze/regen_logs/pack37fc_%A_%a.err

# 37-view pack on the sc-freecpu partition (napoli/visionlab nodes — separate
# physical nodes + separate fairshare from simurgh). NFS-direct read of /simurgh2
# mono (these nodes don't share simurgh5's /scr). Reads pack37_remaining.txt in
# REVERSE so it scans toward the simurgh forward array; the "already 37-packed"
# skip-guard makes any overlap a no-op.
set -u
LIST=${LIST:-/simurgh/u/juze/regen_logs/pack37_remaining.txt}
N=$(grep -c . "$LIST")
# reverse index: task 0 -> last line, task 1 -> second-last, ...
LINE=$((N - SLURM_ARRAY_TASK_ID))
[ "$LINE" -lt 1 ] && { echo "task $SLURM_ARRAY_TASK_ID beyond list ($N)"; exit 0; }
SEQ=$(sed -n "${LINE}p" "$LIST")
[ -z "$SEQ" ] && { echo "no seq at line $LINE"; exit 0; }
echo "=== pack37fc $SLURM_JOB_ID a=$SLURM_ARRAY_TASK_ID line=$LINE seq=$SEQ on $SLURMD_NODENAME ==="

PY=/simurgh2/users/juze/anaconda3/envs/mvbodyfit/bin/python
SCRIPT=/simurgh/u/juze/code/HOIM3_Toolbox/scripts/regen/pack_mono_cache_tar.py
NFS=/simurgh2/datasets/HOI-M3/mhr_mono
PACK=/simurgh2/datasets/HOI-M3/mhr_mono_packed
VIEWS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 34 35 36 37 38 39 41"

# skip if already 37-packed
META="$PACK/$SEQ/person0/meta.npz"
if [ -f "$META" ]; then
  nv=$($PY -c "import numpy as np;print(int(np.load('$META')['views'].shape[0]))" 2>/dev/null)
  [ "$nv" = "37" ] && { echo "[$SEQ] already 37-packed; skip"; exit 0; }
fi

t0=$(date +%s)
$PY $SCRIPT --sequence "$SEQ" --mono_root "$NFS" --output_root "$PACK" --views $VIEWS --workers 16
rc=$?
nv=$($PY -c "import numpy as np;print(int(np.load('$META')['views'].shape[0]))" 2>/dev/null || echo ERR)
echo "$(date) [$SEQ] pack37fc rc=$rc views=$nv total=$(($(date +%s)-t0))s"
exit $rc
