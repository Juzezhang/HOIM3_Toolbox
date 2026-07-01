#!/bin/bash
#SBATCH --job-name=cutie_lo
#SBATCH --partition=sc-loprio
#SBATCH --account=default
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/simurgh/u/juze/regen_logs/cutie_lo_%j.out
#SBATCH --error=/simurgh/u/juze/regen_logs/cutie_lo_%j.err
#SBATCH --requeue
#
# Cutie tracking on sc-loprio (preemptable, but plentiful capacity).
# Same payload as sbatch_cutie_track.sh, different partition + requeue on preempt.
set -eo pipefail
SEQ="${1:?seq required}"
VIEW="${2:?view required}"

CONDA_BASE=/simurgh2/users/juze/anaconda3
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

echo "[cutie_track loprio sbatch] done seq=$SEQ view=$VIEW @ $(date)"
