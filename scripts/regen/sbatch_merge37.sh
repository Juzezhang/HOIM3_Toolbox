#!/bin/bash
#SBATCH --job-name=merge37
#SBATCH --partition=sc-freecpu
#SBATCH --account=simurgh
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=/simurgh/u/juze/regen_logs/merge37_%A_%a.out
#SBATCH --error=/simurgh/u/juze/regen_logs/merge37_%A_%a.err

# Merge an existing CLEAN 16-view pack with its 21 missing canonical views -> 37-view.
# Reads only the missing views (mostly data.tar, cheap) instead of re-reading all 37 ->
# ~half the /simurgh2 NFS reads vs a full repack. Runs on the free CPU pool.
# LIST = clean-16 merge candidates (122 seqs); merge_pack_37.py self-skips if a person
# is incomplete/has .gstmp (those go to full pack instead).
set -u
LIST=${LIST:-/simurgh/u/juze/regen_logs/merge_clean16_list.txt}
SEQ=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$LIST")
[ -z "$SEQ" ] && { echo "no seq at line $((SLURM_ARRAY_TASK_ID+1))"; exit 0; }
echo "=== merge37 $SLURM_JOB_ID a=$SLURM_ARRAY_TASK_ID seq=$SEQ on $SLURMD_NODENAME ==="

PY=/simurgh2/users/juze/anaconda3/envs/mvbodyfit/bin/python
SCRIPT=/simurgh/u/juze/code/HOIM3_Toolbox/scripts/regen/merge_pack_37.py
PACK=/simurgh2/datasets/HOI-M3/mhr_mono_packed

# skip if already 37-packed
META="$PACK/$SEQ/person0/meta.npz"
if [ -f "$META" ]; then
  nv=$($PY -c "import numpy as np;print(int(np.load('$META')['views'].shape[0]))" 2>/dev/null)
  [ "$nv" = "37" ] && { echo "[$SEQ] already 37-packed; skip"; exit 0; }
fi

t0=$(date +%s)
$PY $SCRIPT --sequence "$SEQ" --workers 16
rc=$?
nv=$($PY -c "import numpy as np;print(int(np.load('$META')['views'].shape[0]))" 2>/dev/null || echo ERR)
echo "$(date) [$SEQ] merge37 rc=$rc views=$nv total=$(($(date +%s)-t0))s"
exit $rc
