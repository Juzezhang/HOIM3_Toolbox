#!/bin/bash
#SBATCH --job-name=vitpose
#SBATCH --partition=simurgh
#SBATCH --account=simurgh
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=21-00:00:00
#SBATCH --output=/simurgh/u/juze/regen_logs/vitpose_%A.out
#SBATCH --error=/simurgh/u/juze/regen_logs/vitpose_%A.err
#SBATCH --exclude=simurgh2

# Run ViTPose precompute on HOI-M3 for one sequence.
# After rsync of mono outputs (views 0-15) to /simurgh2 NFS, this job can run
# on ANY simurgh node (no nodelist pinning).
#
# Usage: sbatch sbatch_vitpose.sh <sequence_name>

set -u
SEQ=${1:?seq required}
echo "=== Job $SLURM_JOB_ID on $SLURMD_NODENAME for $SEQ ==="
nvidia-smi -L

source /simurgh2/users/juze/anaconda3/etc/profile.d/conda.sh
conda activate mvbodyfit

PY=/simurgh2/users/juze/anaconda3/envs/mvbodyfit/bin/python
SCRIPT=/simurgh/u/juze/code/mv-bodyfit/tools/precompute_vitpose_hoim3.py

MONO_ROOT=/simurgh2/datasets/HOI-M3/mhr_mono
MASK_SHARD_ROOT=/simurgh2/datasets/HOI-M3/mask_shards
IMAGE_ROOT=/simurgh2/datasets/HOI-M3/images
OUTPUT_ROOT=/simurgh2/datasets/HOI-M3/vitpose_perview

# Sanity: mono must exist for this seq (we're rerunning because masks changed,
# but the script needs at least one mhr_mono view to discover n_frames).
if [ ! -d "$MONO_ROOT/$SEQ" ]; then
    echo "ERROR: $MONO_ROOT/$SEQ missing — no mono data to read bboxes from"
    exit 2
fi

# Wipe stale vitpose output for this seq so re-run starts clean.
# (Script overwrites per-person .npy but we also clear in case persons were renamed.)
rm -rf "$OUTPUT_ROOT/$SEQ"

cd /simurgh/u/juze/code/mv-bodyfit

t0=$(date +%s)
CUDA_VISIBLE_DEVICES=0 $PY $SCRIPT \
    --sequence "$SEQ" \
    --mono_root "$MONO_ROOT" \
    --mask_shard_root "$MASK_SHARD_ROOT" \
    --image_root "$IMAGE_ROOT" \
    --output_root "$OUTPUT_ROOT" \
    --frame_stride 4
rc=$?
dt=$(($(date +%s) - t0))
echo "$(date) [$SEQ] vitpose finished rc=$rc in ${dt}s"
exit $rc
