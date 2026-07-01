#!/bin/bash
#SBATCH --job-name=vitp_swapfix
#SBATCH --partition=simurgh
#SBATCH --account=simurgh
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=21-00:00:00
#SBATCH --output=/simurgh/u/juze/regen_logs/vitp_swapfix_%A.out
#SBATCH --error=/simurgh/u/juze/regen_logs/vitp_swapfix_%A.err
#SBATCH --nodelist=simurgh5

# ViTPose precompute for the 6 new views (17,19,21,22,23,24) on a swap-fix seq.
# Submitted with --dependency=afterok:<6 mono job ids> by the swap-fix cascade
# watcher so this only runs once all 6 mono regen jobs finish.
#
# Usage: sbatch sbatch_vitpose_swapfix.sh <sequence_name>

set -u
SEQ=${1:?seq required}
echo "=== Job $SLURM_JOB_ID on $SLURMD_NODENAME for $SEQ views 17,19,21,22,23,24 ==="
nvidia-smi -L

source /simurgh2/users/juze/anaconda3/etc/profile.d/conda.sh
conda activate mvbodyfit

PY=/simurgh2/users/juze/anaconda3/envs/mvbodyfit/bin/python
SCRIPT=/simurgh/u/juze/code/mv-bodyfit/tools/precompute_vitpose_hoim3.py

MONO_ROOT=/simurgh2/datasets/HOI-M3/mhr_mono
MASK_SHARD_ROOT=/simurgh2/datasets/HOI-M3/mask_shards
IMAGE_ROOT=/simurgh2/datasets/HOI-M3/images
OUTPUT_ROOT=/scr/juze/datasets/HOI-M3/vitpose_new6

if [ ! -d "$MONO_ROOT/$SEQ" ]; then
    echo "ERROR: $MONO_ROOT/$SEQ missing"
    exit 2
fi

cd /simurgh/u/juze/code/mv-bodyfit

t0=$(date +%s)
CUDA_VISIBLE_DEVICES=0 $PY $SCRIPT \
    --sequence "$SEQ" \
    --views 17 19 21 22 23 24 \
    --mono_root "$MONO_ROOT" \
    --mask_shard_root "$MASK_SHARD_ROOT" \
    --image_root "$IMAGE_ROOT" \
    --output_root "$OUTPUT_ROOT"
rc=$?
dt=$(($(date +%s) - t0))
echo "$(date) [$SEQ] vitpose_swapfix finished rc=$rc in ${dt}s"
exit $rc
