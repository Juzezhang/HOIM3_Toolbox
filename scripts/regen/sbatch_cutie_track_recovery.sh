#!/bin/bash
#SBATCH --job-name=cutie_rec
#SBATCH --partition=simurgh
#SBATCH --account=simurgh
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/simurgh/u/juze/regen_logs/cutie_rec_%j.out
#SBATCH --error=/simurgh/u/juze/regen_logs/cutie_rec_%j.err
#SBATCH --exclude=simurgh2

# Cutie tracking recovery for SAM3-anchored missing object.
# Usage: sbatch sbatch_cutie_track_recovery.sh <seq> <view> <obj>

set -u
SEQ=${1:?seq required}
VIEW=${2:?view required}
OBJ=${3:?obj required}
echo "=== Job $SLURM_JOB_ID on $SLURMD_NODENAME — cutie_rec $SEQ v$VIEW obj=$OBJ ==="
nvidia-smi -L

set +u
source /simurgh2/users/juze/anaconda3/etc/profile.d/conda.sh
conda activate HHOI-Toolkit
set -u

REF_ROOT=/simurgh2/datasets/HOI-M3/cutie_refs_recovery_${OBJ}
OUTPUT_ROOT=/simurgh2/datasets/HOI-M3/cutie_tracking_recovery_${OBJ}

cd /simurgh/u/juze/code/HOIM3_Toolbox
python scripts/regen/cutie_track_one_view.py \
    --seq "$SEQ" --view "$VIEW" \
    --ref_root "$REF_ROOT" \
    --video_root /simurgh2/datasets/HOI-M3/videos \
    --output_root "$OUTPUT_ROOT" \
    --max_internal_size 480 \
    --frame_stride 2
echo "$(date) [cutie_rec $SEQ/v$VIEW/$OBJ] done rc=$?"
