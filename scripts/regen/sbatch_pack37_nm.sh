#!/bin/bash
#SBATCH --job-name=pack37nm
#SBATCH --partition=sc-freecpu
#SBATCH --account=simurgh
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/simurgh/u/juze/regen_logs/pack37nm_%A_%a.out
#SBATCH --error=/simurgh/u/juze/regen_logs/pack37nm_%A_%a.err
# Full-pack the NON-mergeable seqs (corrupt 16-view / unpacked / odd view counts) that
# merge_pack_37 can't reuse. Forward-scan fullpack_nonmergeable.txt. NFS-direct tar-aware.
set -u
LIST=${LIST:-/simurgh/u/juze/regen_logs/fullpack_nonmergeable.txt}
SEQ=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$LIST")
[ -z "$SEQ" ] && { echo "no seq at line $((SLURM_ARRAY_TASK_ID+1))"; exit 0; }
echo "=== pack37nm $SLURM_JOB_ID a=$SLURM_ARRAY_TASK_ID seq=$SEQ on $SLURMD_NODENAME ==="
PY=/simurgh2/users/juze/anaconda3/envs/mvbodyfit/bin/python
SCRIPT=/simurgh/u/juze/code/HOIM3_Toolbox/scripts/regen/pack_mono_cache_tar.py
NFS=/simurgh2/datasets/HOI-M3/mhr_mono
PACK=/simurgh2/datasets/HOI-M3/mhr_mono_packed
VIEWS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 34 35 36 37 38 39 41"
META="$PACK/$SEQ/person0/meta.npz"
if [ -f "$META" ]; then nv=$($PY -c "import numpy as np;print(int(np.load('$META')['views'].shape[0]))" 2>/dev/null); [ "$nv" = "37" ] && { echo "[$SEQ] already 37; skip"; exit 0; }; fi
t0=$(date +%s)
$PY $SCRIPT --sequence "$SEQ" --mono_root "$NFS" --output_root "$PACK" --views $VIEWS --workers 16
rc=$?; nv=$($PY -c "import numpy as np;print(int(np.load('$META')['views'].shape[0]))" 2>/dev/null||echo ERR)
echo "$(date) [$SEQ] pack37nm rc=$rc views=$nv total=$(($(date +%s)-t0))s"; exit $rc
