#!/bin/bash
#SBATCH --job-name=meshvizs
#SBATCH --partition=simurgh
#SBATCH --account=simurgh
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --output=/simurgh/u/juze/regen_logs/mesh_viz_skel_%A.out
#SBATCH --error=/simurgh/u/juze/regen_logs/mesh_viz_skel_%A.err
#SBATCH --exclude=simurgh2

# Render HOI-M3 mesh visualization video for the 28-view skel-FK fits
# (mhr_simplified_skel) using pyrender + refined calib.
# Usage: sbatch sbatch_mesh_viz_skel.sh <seq>
set -u
SEQ=${1:?seq required}
echo "=== Job $SLURM_JOB_ID on $SLURMD_NODENAME — mesh_viz_skel $SEQ ==="
nvidia-smi -L

source /simurgh2/users/juze/anaconda3/etc/profile.d/conda.sh
conda activate mvbodyfit

PY=/simurgh2/users/juze/anaconda3/envs/mvbodyfit/bin/python
SCRIPT=/simurgh/u/juze/code/mv-bodyfit/tools/visualize_results_hoim3_video.py
OUT_DIR=/simurgh2/datasets/HOI-M3/viz_mhr_mesh_skel
MHR_ROOT=/simurgh2/datasets/HOI-M3/mhr_simplified_skel

mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/${SEQ}.mp4"

if [ -f "$OUT" ] && [ "$(stat -c %s "$OUT")" -gt 1000000 ]; then
    echo "Skip: $OUT already exists ($(du -sh $OUT | awk '{print $1}'))"
    exit 0
fi

t0=$(date +%s)
PYOPENGL_PLATFORM=egl $PY $SCRIPT \
    --seqs "$SEQ" \
    --mhr_root "$MHR_ROOT" \
    --num_frames 300 \
    --frame_stride 6 \
    --fps 10 \
    --output_dir "$OUT_DIR" \
    --overwrite
rc=$?
dt=$(($(date +%s) - t0))
echo "$(date) [$SEQ] mesh_viz_skel rc=$rc in ${dt}s"
exit $rc
