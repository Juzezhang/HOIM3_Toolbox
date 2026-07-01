#!/bin/bash
#SBATCH --job-name=cutie_trk
#SBATCH --partition=simurgh
#SBATCH --account=simurgh
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/simurgh/u/juze/regen_logs/cutie_%j.out
#SBATCH --error=/simurgh/u/juze/regen_logs/cutie_%j.err
#SBATCH --exclude=simurgh2,simurgh5
#
# Cutie tracking — one (seq, view) per job.
# Usage:  sbatch sbatch_cutie_track.sh <seq> <view>
# Env:    HHOI-Toolkit (cutie + hydra)

set -eo pipefail
SEQ="${1:?seq required}"
VIEW="${2:?view required}"

CONDA_BASE=/simurgh2/users/juze/anaconda3
# Disable -u during conda activate (sets uninitialized vars)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate HHOI-Toolkit
set -u

cd /simurgh/u/juze/code/HOIM3_Toolbox

python scripts/regen/cutie_track_one_view.py \
    --seq "$SEQ" \
    --view "$VIEW" \
    --ref_root /simurgh2/datasets/HOI-M3/cutie_refs \
    --video_root /simurgh2/datasets/HOI-M3/videos \
    --output_root /simurgh2/datasets/HOI-M3/cutie_tracking \
    --max_internal_size 480

echo "[cutie_track sbatch] done seq=$SEQ view=$VIEW @ $(date)"
