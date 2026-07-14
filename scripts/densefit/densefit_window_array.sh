#!/bin/bash
#SBATCH -p sc-freegpu -A default --exclude=orion-dgx
#SBATCH --gres=gpu:1 --cpus-per-task=8 --mem=48G --time=05:00:00
#SBATCH -o /simurgh2/users/juze/calibjoint/densefit_logs/%x_%A_%a.log
source /simurgh2/users/juze/anaconda3/etc/profile.d/conda.sh; conda activate densefit
export CUDA_HOME=/usr/local/cuda-12.4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi -L >/dev/null 2>&1 || { echo "NO_GPU_VISIBLE on $(hostname)"; exit 66; }
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1); [ "${VRAM:-0}" -ge 9000 ] || { echo "VRAM_TOO_SMALL ${VRAM}MiB on $(hostname)"; exit 66; }
python /simurgh2/users/juze/calibjoint/densefit_run_window.py "$SEQ" "$SLURM_ARRAY_TASK_ID" "$WIN"
