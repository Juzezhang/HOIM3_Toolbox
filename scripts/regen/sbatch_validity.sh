#!/bin/bash
#SBATCH --job-name=validity
#SBATCH --partition=simurgh
#SBATCH --account=simurgh
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/simurgh/u/juze/regen_logs/validity_%A.out
#SBATCH --error=/simurgh/u/juze/regen_logs/validity_%A.err
#SBATCH --exclude=simurgh2

# Regen mask validity for one seq (GPU-batched multi-view voxel hull check).
# Usage: sbatch sbatch_validity.sh <seq>
set -u
SEQ=${1:?seq required}
echo "=== Job $SLURM_JOB_ID on $SLURMD_NODENAME — validity $SEQ ==="
nvidia-smi -L

source /simurgh2/users/juze/anaconda3/etc/profile.d/conda.sh
conda activate HOIM3_Toolbox

PY=/simurgh2/users/juze/anaconda3/envs/HOIM3_Toolbox/bin/python
$PY /simurgh/u/juze/code/HOIM3_Toolbox/scripts/multi_view_mask_check.py \
    --root_path /simurgh2/datasets/HOI-M3 \
    --seq_name "$SEQ" \
    --output_path /scr/juze/datasets/HOI-M3/mask_validity \
    --all_views \
    --device cuda
rc=$?
echo "$(date) [$SEQ] validity rc=$rc"
exit $rc
