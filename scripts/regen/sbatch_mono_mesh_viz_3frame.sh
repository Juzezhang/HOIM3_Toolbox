#!/bin/bash
#SBATCH --job-name=monomesh3
#SBATCH --partition=simurgh
#SBATCH --account=simurgh
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --exclude=simurgh2
#SBATCH --output=/simurgh/u/juze/regen_logs/monomesh3_%j.out
#SBATCH --error=/simurgh/u/juze/regen_logs/monomesh3_%j.err
#SBATCH --requeue
#
# Render mono-MHR mesh on a 28-view x 3-frame composite PNG per HOI-M3 sequence.
# Output: /simurgh2/datasets/HOI-M3/viz_mono_mesh_3frame/<seq>.png
set -eo pipefail
SEQ="${1:?seq required}"

CONDA_BASE=/simurgh2/users/juze/anaconda3
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate mvbodyfit
set -u

cd /simurgh/u/juze/code/mv-bodyfit

python tools/visualize_mono_mhr_3frame_28v.py \
    --sequences "$SEQ" \
    --out_dir /simurgh2/datasets/HOI-M3/viz_mono_mesh_3frame \
    --gpu 0

echo "[monomesh3 sbatch] done seq=$SEQ @ $(date)"
