#!/bin/bash
#SBATCH --job-name=pack37tar
#SBATCH --partition=simurgh
#SBATCH --account=simurgh
#SBATCH --cpus-per-task=20
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=/simurgh/u/juze/regen_logs/pack37tar_%A_%a.out
#SBATCH --error=/simurgh/u/juze/regen_logs/pack37tar_%A_%a.err

# Tar-aware 37-view pack of one seq (line $SLURM_ARRAY_TASK_ID of $LIST).
# CPU-only (no GPU). Reads loose npz + data.tar from NFS, writes packed cache.
set -u
LIST=${LIST:?set LIST}
SEQ=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$LIST")
[ -z "$SEQ" ] && { echo "no seq at line $((SLURM_ARRAY_TASK_ID+1))"; exit 0; }
echo "=== pack37tar $SLURM_JOB_ID array=$SLURM_ARRAY_TASK_ID seq=$SEQ on $SLURMD_NODENAME ==="

PY=/simurgh2/users/juze/anaconda3/envs/mvbodyfit/bin/python
SCRIPT=/simurgh/u/juze/code/HOIM3_Toolbox/scripts/regen/pack_mono_cache_tar.py
PACK=/simurgh2/datasets/HOI-M3/mhr_mono_packed
VIEWS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 34 35 36 37 38 39 41"

# skip if already 37-packed
META="$PACK/$SEQ/person0/meta.npz"
if [ -f "$META" ]; then
  nv=$($PY -c "import numpy as np;print(int(np.load('$META')['views'].shape[0]))" 2>/dev/null)
  [ "$nv" = "37" ] && { echo "[$SEQ] already 37-packed; skip"; exit 0; }
fi

t0=$(date +%s)
$PY $SCRIPT --sequence "$SEQ" \
  --mono_root /simurgh2/datasets/HOI-M3/mhr_mono \
  --output_root "$PACK" --views $VIEWS --workers 20
rc=$?
nv=$($PY -c "import numpy as np;print(int(np.load('$META')['views'].shape[0]))" 2>/dev/null || echo ERR)
echo "$(date) [$SEQ] pack37tar rc=$rc views=$nv $(($(date +%s)-t0))s"
exit $rc
