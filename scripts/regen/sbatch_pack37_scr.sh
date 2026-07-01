#!/bin/bash
#SBATCH --job-name=pack37scr
#SBATCH --partition=simurgh
#SBATCH --account=simurgh
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=/simurgh/u/juze/regen_logs/pack37scr_%A_%a.out
#SBATCH --error=/simurgh/u/juze/regen_logs/pack37scr_%A_%a.err

# 37-view pack with NODE-LOCAL /scr staging to avoid /simurgh2 NFS read contention.
# Stage the seq's mono (loose npz + data.tar) to node-local /scr, extract tars there,
# pack reading from /scr, write packed cache to /simurgh2, then clean /scr.
set -u
LIST=${LIST:?set LIST}
SEQ=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$LIST")
[ -z "$SEQ" ] && { echo "no seq at line $((SLURM_ARRAY_TASK_ID+1))"; exit 0; }
echo "=== pack37scr $SLURM_JOB_ID a=$SLURM_ARRAY_TASK_ID seq=$SEQ on $SLURMD_NODENAME ==="

PY=/simurgh2/users/juze/anaconda3/envs/mvbodyfit/bin/python
SCRIPT=/simurgh/u/juze/code/HOIM3_Toolbox/scripts/regen/pack_mono_cache_tar.py
NFS=/simurgh2/datasets/HOI-M3/mhr_mono
PACK=/simurgh2/datasets/HOI-M3/mhr_mono_packed
SCRBASE=/scr/juze/pack_stage/$SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID
VIEWS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 34 35 36 37 38 39 41"

# skip if already 37-packed
META="$PACK/$SEQ/person0/meta.npz"
if [ -f "$META" ]; then
  nv=$($PY -c "import numpy as np;print(int(np.load('$META')['views'].shape[0]))" 2>/dev/null)
  [ "$nv" = "37" ] && { echo "[$SEQ] already 37-packed; skip"; exit 0; }
fi

t0=$(date +%s)
mkdir -p "$SCRBASE/$SEQ"
# stage: copy the whole seq mono dir to node-local /scr (one big sequential read from NFS)
echo "[$SEQ] staging NFS -> $SCRBASE (cp -r)"
cp -r "$NFS/$SEQ/." "$SCRBASE/$SEQ/" 2>/dev/null
t1=$(date +%s); echo "[$SEQ] stage took $((t1-t0))s"
# extract any data.tar on local /scr (fast local SSD)
find "$SCRBASE/$SEQ" -maxdepth 2 -name data.tar | xargs -r -P 12 -I {} bash -c 'd=$(dirname "{}"); tar -xf "{}" -C "$d/" && rm -f "{}"'
t2=$(date +%s); echo "[$SEQ] extract took $((t2-t1))s"
# pack reading from /scr
$PY $SCRIPT --sequence "$SEQ" --mono_root "$SCRBASE" --output_root "$PACK" --views $VIEWS --workers 20
rc=$?
t3=$(date +%s)
nv=$($PY -c "import numpy as np;print(int(np.load('$META')['views'].shape[0]))" 2>/dev/null || echo ERR)
echo "$(date) [$SEQ] pack37scr rc=$rc views=$nv stage=$((t1-t0))s extract=$((t2-t1))s pack=$((t3-t2))s total=$((t3-t0))s"
# cleanup local /scr
rm -rf "$SCRBASE"
exit $rc
