#!/bin/bash
#SBATCH --job-name=fit37v2sh
#SBATCH --partition=simurgh
#SBATCH --account=simurgh
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/simurgh/u/juze/regen_logs/fit_v2_shape_only_%A.out
#SBATCH --error=/simurgh/u/juze/regen_logs/fit_v2_shape_only_%A.err
#SBATCH --exclude=simurgh2

# v2 SHAPE-ONLY (identity-only) ablation. Clone of sbatch_fit_simplified_skel_v2.sh;
# only differences: top-level cfg + OUT_DIR.
# Usage: sbatch sbatch_fit_v2_shape_only.sh <seq>

set -u
SEQ=${1:?seq required}
echo "=== Job $SLURM_JOB_ID on $SLURMD_NODENAME — fit_v2_shape_only $SEQ ==="
nvidia-smi -L

source /simurgh2/users/juze/anaconda3/etc/profile.d/conda.sh
conda activate mvbodyfit

PY=/simurgh2/users/juze/anaconda3/envs/mvbodyfit/bin/python
OUT_DIR=/simurgh2/datasets/HOI-M3/mhr_simplified_skel_v2_shape_only/$SEQ

if [ -d "$OUT_DIR/keypoints3d" ] && [ "$(/usr/bin/ls $OUT_DIR/keypoints3d 2>/dev/null | /usr/bin/wc -l)" -ge 100 ] \
   && [ -d "$OUT_DIR/mhr" ] && [ "$(/usr/bin/ls $OUT_DIR/mhr 2>/dev/null | /usr/bin/wc -l)" -ge 100 ]; then
    echo "Skip: $OUT_DIR already has keypoints3d+mhr"
    exit 0
fi

cd /simurgh/u/juze/code/mv-bodyfit

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MVFIT_GLOBAL_SHAPE_N=50
export MVFIT_CHUNK_SIZE=21787
export CUDA_VISIBLE_DEVICES=0

# 37 views — exclude v30/v31/v32/v33/v40 (calibration/missing)
SUBS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 34 35 36 37 38 39 41"

t0=$(date +%s)
$PY -u apps/mocap/run.py \
    --cfg config/hoim3_mhr_simplified_skel_v2_shape_only_refined.yml \
    --sequence "$SEQ" \
    --subs $SUBS \
    --out "$OUT_DIR" \
    --skip_vis
rc=$?
dt=$(($(date +%s) - t0))
echo "$(date) [$SEQ] fit_v2_shape_only rc=$rc in ${dt}s"
exit $rc
