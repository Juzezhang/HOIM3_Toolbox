#!/bin/bash
#SBATCH --job-name=fit
#SBATCH --partition=simurgh
#SBATCH --account=simurgh
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/simurgh/u/juze/regen_logs/fit_%A.out
#SBATCH --error=/simurgh/u/juze/regen_logs/fit_%A.err
#SBATCH --exclude=simurgh2

# Run mvbodyfit MHR Simplified Refined fitting for one HOI-M3 sequence.
# Uses 17 views (canonical 10 + supplementary 17/19/21/22/23/24/25).
# Output: /simurgh2/datasets/HOI-M3/mhr_simplified/<seq>/
#
# Usage:  sbatch sbatch_fit.sh <seq>

set -u
SEQ=${1:?seq required}
echo "=== Job $SLURM_JOB_ID on $SLURMD_NODENAME — fit $SEQ ==="
nvidia-smi -L

source /simurgh2/users/juze/anaconda3/etc/profile.d/conda.sh
conda activate mvbodyfit

PY=/simurgh2/users/juze/anaconda3/envs/mvbodyfit/bin/python
OUT_DIR=/simurgh2/datasets/HOI-M3/mhr_simplified/$SEQ

# Skip if already complete — REQUIRE BOTH keypoints3d AND mhr (per-frame JSON params).
# Old fit runs produced only keypoints3d/, missing mhr/. Mesh viz needs mhr/.
if [ -d "$OUT_DIR/keypoints3d" ] && [ "$(/usr/bin/ls $OUT_DIR/keypoints3d 2>/dev/null | /usr/bin/wc -l)" -ge 100 ] \
   && [ -d "$OUT_DIR/mhr" ] && [ "$(/usr/bin/ls $OUT_DIR/mhr 2>/dev/null | /usr/bin/wc -l)" -ge 100 ]; then
    echo "Skip: $OUT_DIR has both keypoints3d ($(/usr/bin/ls $OUT_DIR/keypoints3d | /usr/bin/wc -l)) + mhr ($(/usr/bin/ls $OUT_DIR/mhr | /usr/bin/wc -l)) frames"
    exit 0
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
echo "$(date) [$SEQ] fit rc=$rc in ${dt}s"
exit $rc
