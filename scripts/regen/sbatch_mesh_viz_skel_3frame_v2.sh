#!/bin/bash
#SBATCH --job-name=meshviz3v2
#SBATCH --partition=simurgh
#SBATCH --account=simurgh
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:10:00
#SBATCH --output=/simurgh/u/juze/regen_logs/mesh_viz_skel_3frame_v2_%A.out
#SBATCH --error=/simurgh/u/juze/regen_logs/mesh_viz_skel_3frame_v2_%A.err
#SBATCH --exclude=simurgh2

# 3-frame (first/middle/last) composite PNG renderer for HOI-M3 mesh-skel-V2 viz.
# Clone of sbatch_mesh_viz_skel_3frame.sh; only changes the input MHR_ROOT and
# OUT_DIR to v2.
# Usage: sbatch sbatch_mesh_viz_skel_3frame_v2.sh <seq>
set -u
SEQ=${1:?seq required}
echo "=== Job $SLURM_JOB_ID on $SLURMD_NODENAME — mesh_viz_skel_3frame_v2 $SEQ ==="
nvidia-smi -L

source /simurgh2/users/juze/anaconda3/etc/profile.d/conda.sh
conda activate mvbodyfit

PY=/simurgh2/users/juze/anaconda3/envs/mvbodyfit/bin/python
SCRIPT=/simurgh/u/juze/code/mv-bodyfit/tools/visualize_results_hoim3_3frame.py
OUT_DIR=/simurgh2/datasets/HOI-M3/viz_mhr_mesh_skel_3frame_v2
MHR_ROOT=/simurgh2/datasets/HOI-M3/mhr_simplified_skel_v2

mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/${SEQ}.png"

if [ -f "$OUT" ] && [ "$(stat -c %s "$OUT")" -gt 50000 ]; then
    echo "Skip: $OUT already exists ($(du -sh $OUT | awk '{print $1}'))"
    exit 0
fi

t0=$(date +%s)
PYOPENGL_PLATFORM=egl $PY $SCRIPT \
    --seqs "$SEQ" \
    --mhr_root "$MHR_ROOT" \
    --output_dir "$OUT_DIR" \
    --overwrite
rc=$?
dt=$(($(date +%s) - t0))
echo "$(date) [$SEQ] mesh_viz_skel_3frame_v2 rc=$rc in ${dt}s"
exit $rc
