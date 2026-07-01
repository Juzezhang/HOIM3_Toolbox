#!/bin/bash
#SBATCH --job-name=sam3_probe
#SBATCH --partition=simurgh
#SBATCH --account=simurgh
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/simurgh/u/juze/regen_logs/sam3_probe_%A.out
#SBATCH --error=/simurgh/u/juze/regen_logs/sam3_probe_%A.err
#SBATCH --exclude=simurgh2

set -u
echo "=== Job $SLURM_JOB_ID on $SLURMD_NODENAME — sam3_probe ==="
nvidia-smi -L

source /simurgh2/users/juze/anaconda3/etc/profile.d/conda.sh
conda activate sam3

cd /simurgh/u/juze/code/sam3
/simurgh2/users/juze/anaconda3/envs/sam3/bin/python /simurgh/u/juze/code/HOIM3_Toolbox/scripts/regen/sam3_detect_missing_obj.py
rc=$?
echo "$(date) rc=$rc"
exit $rc
