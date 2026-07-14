#!/bin/bash
#SBATCH --job-name=cutie_mt
#SBATCH --partition=sc-freegpu
#SBATCH --account=default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/simurgh/u/juze/regen_logs/mt_%x_%A_%a.out
#SBATCH --error=/simurgh/u/juze/regen_logs/mt_%x_%A_%a.err
# Full-length Cutie tracking, one (seq,view) per array task. SEQ from --export.
# 42 shard views = array 0-41. NON-DESTRUCTIVE: writes cutie_tracking/<seq>/<view>/.
set -eo pipefail
: "${SEQ:?SEQ required via --export}"
VIEW=$SLURM_ARRAY_TASK_ID           # views are 0..41 == array index
# skip if already done
DONE=/simurgh2/datasets/HOI-M3/cutie_tracking/$SEQ/$VIEW/.tracked_done
if [ -f "$DONE" ]; then echo "[skip] $SEQ v$VIEW already tracked"; exit 0; fi
source /simurgh2/users/juze/anaconda3/etc/profile.d/conda.sh
conda activate HHOI-Toolkit
set -u
cd /simurgh/u/juze/code/HOIM3_Toolbox
python scripts/regen/cutie_track_one_view.py \
    --seq "$SEQ" --view "$VIEW" \
    --ref_root /simurgh2/datasets/HOI-M3/cutie_refs \
    --video_root /simurgh2/datasets/HOI-M3/videos \
    --output_root /simurgh2/datasets/HOI-M3/cutie_tracking \
    --max_internal_size 480
echo "[done] $SEQ v$VIEW @ $(date)"
