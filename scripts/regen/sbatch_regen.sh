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

set -u
echo "=== Job $SLURM_JOB_ID on $SLURMD_NODENAME ==="
nvidia-smi -L

PY=/simurgh2/users/juze/anaconda3/envs/HOIM3_Toolbox/bin/python
INFER=/simurgh/u/juze/code/HOIM3_Toolbox/scripts/yolo_seg/inference_masks.py
FIX=/simurgh/u/juze/code/HOIM3_Toolbox/scripts/regen/fix_one_seq_v2.sh

# Take seq from arg
SEQ=${1:?seq name required}

echo "$(date) [$SEQ] STAGE 1: inference"
$PY $INFER --gpu 0 --sequences $SEQ
rc=$?
[ $rc -ne 0 ] && { echo "inference FAILED rc=$rc"; exit 1; }

echo "$(date) [$SEQ] STAGE 2: fix"
bash $FIX $SEQ
rc=$?
[ $rc -ne 0 ] && { echo "fix FAILED rc=$rc"; exit 1; }

echo "$(date) [$SEQ] DONE"
