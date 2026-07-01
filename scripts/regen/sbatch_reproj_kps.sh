#!/bin/bash
#SBATCH --job-name=kp_reproj
#SBATCH --partition=simurgh
#SBATCH --account=simurgh
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/simurgh/u/juze/regen_logs/kp_reproj_%A_%j.out
#SBATCH --error=/simurgh/u/juze/regen_logs/kp_reproj_%A_%j.err
# NOTE: NO --gres=gpu -- pure CPU job (cv2 + ffmpeg).

set -u
SEQ=${1:?seq name required}
echo "=== Job $SLURM_JOB_ID on $SLURMD_NODENAME  seq=$SEQ ==="
date
PY=/simurgh2/users/juze/anaconda3/envs/HOIM3_Toolbox/bin/python
SCRIPT=/simurgh/u/juze/code/HOIM3_Toolbox/scripts/regen/reproj_kps_to_grid_video.py
$PY $SCRIPT --seq "$SEQ"
echo "=== done $(date) ==="
