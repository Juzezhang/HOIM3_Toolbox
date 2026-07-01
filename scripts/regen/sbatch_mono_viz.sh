#!/bin/bash
#SBATCH --job-name=mono_viz
#SBATCH --partition=simurgh
#SBATCH --account=simurgh
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=8:00:00
#SBATCH --output=/simurgh/u/juze/regen_logs/mono_viz_%A.out
#SBATCH --error=/simurgh/u/juze/regen_logs/mono_viz_%A.err
#SBATCH --exclude=simurgh2

# Render 10-view MHR-mono mesh-overlay grid mp4 for one sequence.
# Matches existing /simurgh2/datasets/HOI-M3/mono_viz/mono_viz_bedroom_data01_grid_10fps.mp4 style.
# Usage: sbatch sbatch_mono_viz.sh <seq>

set -u
SEQ=${1:?seq required}
echo "=== Job $SLURM_JOB_ID on $SLURMD_NODENAME — mono_viz $SEQ ==="
nvidia-smi -L

source /simurgh2/users/juze/anaconda3/etc/profile.d/conda.sh
conda activate mvbodyfit
PY=/simurgh2/users/juze/anaconda3/envs/mvbodyfit/bin/python

OUT_DIR=/simurgh2/datasets/HOI-M3/mono_viz
OUT_FILE="$OUT_DIR/mono_viz_${SEQ}_grid_10fps.mp4"
mkdir -p "$OUT_DIR"

if [ -f "$OUT_FILE" ]; then
    sz=$(/usr/bin/stat -c %s "$OUT_FILE")
    if [ "$sz" -gt 1000000 ]; then
        echo "Skip: $OUT_FILE exists ($sz bytes)"; exit 0
    fi
fi

cd /simurgh/u/juze/code/mv-bodyfit
export PYOPENGL_PLATFORM=egl
export SAM3D_BODY_CKPT=/simurgh/u/juze/code/fast-sam-3d-body/checkpoints/sam-3d-body-dinov3/model.ckpt

# Sample 500 frames evenly across the sequence (~50 sec @ 10 fps).
# Stride is sized so 500 * stride covers most of the seq.
SEQ_FRAMES=$(/usr/bin/ls /simurgh2/datasets/HOI-M3/images/$SEQ/0 2>/dev/null | /usr/bin/grep -c "\.jpg$")
[ "$SEQ_FRAMES" -lt 1 ] && SEQ_FRAMES=21000
STRIDE=$((SEQ_FRAMES / 500))
[ "$STRIDE" -lt 1 ] && STRIDE=1

t0=$(date +%s)
$PY tools/visualize_mono_mhr_mesh.py \
    --sequence "$SEQ" \
    --mono_root /simurgh2/datasets/HOI-M3/mhr_mono \
    --image_root /simurgh2/datasets/HOI-M3/images \
    --views 0 2 5 6 7 8 10 11 14 15 \
    --start_frame 0 --max_frames 500 --stride "$STRIDE" \
    --fps 10 --n_cols 5 \
    --output "$OUT_FILE"
rc=$?
dt=$(($(date +%s) - t0))
echo "$(date) [$SEQ] mono_viz rc=$rc in ${dt}s -> $OUT_FILE"
exit $rc
