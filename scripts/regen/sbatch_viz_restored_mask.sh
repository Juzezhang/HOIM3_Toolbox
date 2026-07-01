#!/bin/bash
#SBATCH --job-name=viz_mask
#SBATCH --partition=simurgh
#SBATCH --account=simurgh
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1:00:00
#SBATCH --output=/simurgh/u/juze/regen_logs/viz_mask_%A.out
#SBATCH --error=/simurgh/u/juze/regen_logs/viz_mask_%A.err
#SBATCH --exclude=simurgh2

# Render 42-view mask overlay grid mp4 (30 sampled frames @ 1fps).
# Output: /simurgh2/datasets/HOI-M3/mask_viz/viz_restored_<seq>.mp4
#
# Usage: sbatch sbatch_viz_restored_mask.sh <seq>

set -u
SEQ=${1:?seq required}
echo "=== Job $SLURM_JOB_ID on $SLURMD_NODENAME — viz_mask $SEQ ==="

source /simurgh2/users/juze/anaconda3/etc/profile.d/conda.sh
conda activate HOIM3_Toolbox
PY=/simurgh2/users/juze/anaconda3/envs/HOIM3_Toolbox/bin/python

OUT_DIR=/simurgh2/datasets/HOI-M3/mask_viz
mkdir -p "$OUT_DIR"
OUT_FILE="$OUT_DIR/viz_restored_${SEQ}.mp4"

t0=$(date +%s)
$PY /simurgh/u/juze/code/HOIM3_Toolbox/scripts/regen/viz_restored_mask.py "$SEQ" 30 "$OUT_FILE"
rc=$?
dt=$(($(date +%s) - t0))
echo "$(date) [$SEQ] viz_mask rc=$rc in ${dt}s -> $OUT_FILE"
exit $rc
