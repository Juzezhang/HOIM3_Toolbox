#!/bin/bash
#SBATCH --job-name=fit_lo
#SBATCH --partition=sc-loprio
#SBATCH --account=default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/simurgh/u/juze/regen_logs/fit_lo_%A.out
#SBATCH --error=/simurgh/u/juze/regen_logs/fit_lo_%A.err
#SBATCH --requeue

# mvbodyfit MHR Simplified Refined fitting on sc-loprio (preemptable).
# Usage:  sbatch sbatch_fit_loprio.sh <seq>

set -u
SEQ=${1:?seq required}
echo "=== Job $SLURM_JOB_ID on $SLURMD_NODENAME — fit_lo $SEQ ==="
nvidia-smi -L

source /simurgh2/users/juze/anaconda3/etc/profile.d/conda.sh
conda activate mvbodyfit

PY=/simurgh2/users/juze/anaconda3/envs/mvbodyfit/bin/python
OUT_DIR=/simurgh2/datasets/HOI-M3/mhr_simplified/$SEQ

if [ -d "$OUT_DIR/keypoints3d" ] && [ "$(/usr/bin/ls $OUT_DIR/keypoints3d 2>/dev/null | /usr/bin/wc -l)" -ge 100 ]; then
    echo "Skip: already done"; exit 0
fi

cd /simurgh/u/juze/code/mv-bodyfit
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MVFIT_CHUNK_SIZE=3000
export CUDA_VISIBLE_DEVICES=0

t0=$(date +%s)
$PY -u apps/mocap/run.py \
    --cfg config/hoim3_mhr_simplified_refined.yml \
    --sequence "$SEQ" \
    --subs 0 2 5 6 7 8 10 11 14 15 17 19 21 22 23 24 \
    --out "$OUT_DIR" \
    --skip_vis
rc=$?
dt=$(($(date +%s) - t0))
echo "$(date) [$SEQ] fit_lo rc=$rc in ${dt}s"
exit $rc
