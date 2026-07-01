#!/bin/bash
#SBATCH --job-name=mask_regen
#SBATCH --partition=simurgh
#SBATCH --account=simurgh
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=21-00:00:00
#SBATCH --output=/simurgh/u/juze/regen_logs/regen_%A_%a.out
#SBATCH --error=/simurgh/u/juze/regen_logs/regen_%A_%a.err
#SBATCH --exclude=simurgh2

set -u
echo "=== Job $SLURM_JOB_ID on $SLURMD_NODENAME ==="
nvidia-smi -L
PY=/simurgh2/users/juze/anaconda3/envs/HOIM3_Toolbox/bin/python
INFER=/simurgh/u/juze/code/HOIM3_Toolbox/scripts/yolo_seg/inference_masks.py
SEQ=${1:?seq name required}

echo "$(date) [$SEQ] inference (NPZ to /simurgh2/datasets/HOI-M3/mask_npz_generated/$SEQ/)"
$PY $INFER --gpu 0 --sequences $SEQ
rc=$?
[ $rc -ne 0 ] && { echo "inference FAILED rc=$rc"; exit 1; }

# Drop a marker so simurgh5 fix_dispatcher knows this seq's NPZ is ready
touch /simurgh2/datasets/HOI-M3/mask_npz_generated/$SEQ/.inference_done
echo "$(date) [$SEQ] inference DONE — marker dropped"
