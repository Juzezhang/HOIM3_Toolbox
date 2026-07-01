#!/bin/bash
#SBATCH --job-name=fitsvd
#SBATCH --partition=simurgh
#SBATCH --account=simurgh
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --output=/simurgh/u/juze/regen_logs/fitsvd_%A.out
#SBATCH --error=/simurgh/u/juze/regen_logs/fitsvd_%A.err
#SBATCH --exclude=simurgh2

# 28-view mvbodyfit with closed-form Kabsch R/T + temporal smoothing (mhr_rt_svd).
# Expected ~3-5 min/seq vs ~50 min for mhr_simplified.
# Usage: sbatch sbatch_fit_rt_svd.sh <seq>

set -u
SEQ=${1:?seq required}
echo "=== Job $SLURM_JOB_ID on $SLURMD_NODENAME — fit_rt_svd $SEQ ==="
nvidia-smi -L

source /simurgh2/users/juze/anaconda3/etc/profile.d/conda.sh
conda activate mvbodyfit

PY=/simurgh2/users/juze/anaconda3/envs/mvbodyfit/bin/python
OUT_DIR=/simurgh2/datasets/HOI-M3/mhr_rt_svd/$SEQ

if [ -d "$OUT_DIR/keypoints3d" ] && [ "$(/usr/bin/ls $OUT_DIR/keypoints3d 2>/dev/null | /usr/bin/wc -l)" -ge 100 ] \
   && [ -d "$OUT_DIR/mhr" ] && [ "$(/usr/bin/ls $OUT_DIR/mhr 2>/dev/null | /usr/bin/wc -l)" -ge 100 ]; then
    echo "Skip: $OUT_DIR already has keypoints3d+mhr"
    exit 0
fi

cd /simurgh/u/juze/code/mv-bodyfit

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Larger chunk size — Kabsch is O(F) and memory-cheap, no LBFGS history.
export MVFIT_CHUNK_SIZE=3000
# Disable subsample+warm-refine path: with closed-form Kabsch per frame there's
# no benefit to fitting a 25-frame subsample and SLERP-interpolating; run the
# analytical solve on every frame instead.
export MVFIT_SUBSAMPLE_STEP=0
export CUDA_VISIBLE_DEVICES=0

# 28 views (canonical 16 + new 6 + 35-41) — exclude v33
SUBS="0 2 5 6 7 8 10 11 14 15 17 19 21 22 23 24 25 26 27 28 29 34 35 36 37 38 39 41"

t0=$(date +%s)
$PY -u apps/mocap/run.py \
    --cfg config/hoim3_mhr_rt_svd_refined.yml \
    --sequence "$SEQ" \
    --subs $SUBS \
    --out "$OUT_DIR" \
    --skip_vis
rc=$?
dt=$(($(date +%s) - t0))
echo "$(date) [$SEQ] fit_rt_svd rc=$rc in ${dt}s"
exit $rc
