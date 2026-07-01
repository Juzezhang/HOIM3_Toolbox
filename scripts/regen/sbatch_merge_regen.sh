#!/bin/bash
#SBATCH --job-name=merge_regen
#SBATCH --partition=simurgh
#SBATCH --account=simurgh
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --exclude=simurgh2
#SBATCH --output=/simurgh/u/juze/regen_logs/merge_regen_%A_%x.out
#SBATCH --error=/simurgh/u/juze/regen_logs/merge_regen_%A_%x.err

# Usage:
#   sbatch sbatch_merge_regen.sh <sequence> [extra args]
# Example:
#   sbatch sbatch_merge_regen.sh bedroom_data01 --refresh_validity
set -eu

SEQ=${1:?"sequence name required"}
shift || true

echo "=== merge_regen $SLURM_JOB_ID on $SLURMD_NODENAME for $SEQ ==="
echo "extra args: $*"
date

# Conda env via direct python binary (avoids `conda activate` shell init issues)
PY=/simurgh2/users/juze/anaconda3/envs/HOIM3_Toolbox/bin/python
SCRIPT=/simurgh/u/juze/code/HOIM3_Toolbox/scripts/regen/merge_regen_mask.py

mkdir -p /simurgh/u/juze/regen_logs

$PY $SCRIPT \
    --sequence "$SEQ" \
    --mask_shard_root /simurgh2/datasets/HOI-M3/mask_shards \
    --generated_root /simurgh2/datasets/HOI-M3/mask_npz_generated \
    --reuse_npz_generated \
    --device cuda \
    "$@"

echo "=== merge_regen $SEQ DONE rc=$? ==="
date
