#!/bin/bash
#SBATCH --job-name=sam3_xv
#SBATCH --partition=simurgh
#SBATCH --account=simurgh
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=/simurgh/u/juze/regen_logs/sam3_xv_%A.out
#SBATCH --error=/simurgh/u/juze/regen_logs/sam3_xv_%A.err
#SBATCH --exclude=simurgh2

# SAM3 cross-view mask propagation for one (seq, obj, ref_view, ref_frame).
# Usage: sbatch sbatch_sam3_xv.sh <seq> <obj> <ref_view> <ref_frame>

set -u
SEQ=${1:?seq required}
OBJ=${2:?obj required}
REFVIEW=${3:?ref_view required}
REFFRAME=${4:?ref_frame required}
echo "=== Job $SLURM_JOB_ID on $SLURMD_NODENAME — sam3_xv $SEQ $OBJ v$REFVIEW f$REFFRAME ==="
nvidia-smi -L

source /simurgh2/users/juze/anaconda3/etc/profile.d/conda.sh
conda activate sam3

cd /simurgh/u/juze/code/sam3
/simurgh2/users/juze/anaconda3/envs/sam3/bin/python \
    /simurgh/u/juze/code/HOIM3_Toolbox/scripts/regen/sam3_crossview_propagate.py \
    --seq "$SEQ" --obj "$OBJ" --ref_view "$REFVIEW" --ref_frame "$REFFRAME"
echo "$(date) rc=$?"
