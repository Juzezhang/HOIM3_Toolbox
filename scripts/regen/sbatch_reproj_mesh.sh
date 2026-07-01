#!/bin/bash
#SBATCH --job-name=mesh_reproj
#SBATCH --partition=simurgh
#SBATCH --account=simurgh
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --output=/simurgh/u/juze/regen_logs/mesh_reproj_%A.out
#SBATCH --error=/simurgh/u/juze/regen_logs/mesh_reproj_%A.err
#SBATCH --exclude=simurgh2

# Render 42-view MHR mesh grid mp4 for one fitted seq.
# Output: /simurgh2/datasets/HOI-M3/mesh_reproj_viz/<seq>.mp4

set -u
SEQ=${1:?seq required}
echo "=== Job $SLURM_JOB_ID on $SLURMD_NODENAME — mesh_reproj $SEQ ==="
nvidia-smi -L

source /simurgh2/users/juze/anaconda3/etc/profile.d/conda.sh
conda activate mvbodyfit
PY=/simurgh2/users/juze/anaconda3/envs/mvbodyfit/bin/python

export PYOPENGL_PLATFORM=egl

cd /simurgh/u/juze/code/mv-bodyfit
t0=$(date +%s)
$PY /simurgh/u/juze/code/HOIM3_Toolbox/scripts/regen/reproj_mesh_to_grid_video.py --seq "$SEQ"
rc=$?
dt=$(($(date +%s) - t0))
echo "$(date) [$SEQ] mesh_reproj rc=$rc in ${dt}s"
exit $rc
