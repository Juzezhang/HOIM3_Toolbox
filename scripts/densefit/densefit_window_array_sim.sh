#!/bin/bash
#SBATCH -p simurgh -A simurgh --gres=gpu:l40s:1 --exclude=simurgh2,simurgh5
#SBATCH --cpus-per-task=8 --mem=48G --time=02:00:00
#SBATCH -o /simurgh2/users/juze/calibjoint/densefit_logs/%x_%A_%a.log
source /simurgh2/users/juze/anaconda3/etc/profile.d/conda.sh; conda activate densefit
export CUDA_HOME=/usr/local/cuda-12.4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python /simurgh2/users/juze/calibjoint/densefit_run_window.py "$SEQ" "$SLURM_ARRAY_TASK_ID" "$WIN"
