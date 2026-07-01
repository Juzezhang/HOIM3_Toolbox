#!/bin/bash
#SBATCH --job-name=vitp_b10_v17v19
#SBATCH --partition=simurgh
#SBATCH --account=simurgh
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=21-00:00:00
#SBATCH --output=/simurgh/u/juze/regen_logs/vitp_b10_v17v19_%A.out
#SBATCH --error=/simurgh/u/juze/regen_logs/vitp_b10_v17v19_%A.err
#SBATCH --nodelist=simurgh5

# Run ViTPose precompute on HOI-M3 bedroom_data10 for views 17 and 19 only.
# These are the gap views where mono was crashed-partial; this job is intended
# to be submitted with --dependency=afterok:<mono_v17_jobid>:<mono_v19_jobid>
# so it kicks off ONLY after both mono regen jobs complete.
#
# Output goes to /scr/juze/datasets/HOI-M3/vitpose_new6 (separate from legacy
# 10-view cache); a downstream merge script can union the new views into the
# legacy keypoints_coco23.npy.
#
# Pinned to simurgh5 because mono outputs land on /scr (simurgh5-local) AFTER
# /simurgh2 -> /scr sync; the mono sbatch wrappers write to /simurgh2 NFS.
# We point MONO_ROOT at /simurgh2 to be safe.

set -u
SEQ=bedroom_data10
echo "=== Job $SLURM_JOB_ID on $SLURMD_NODENAME for $SEQ views 17,19 ==="
nvidia-smi -L

source /simurgh2/users/juze/anaconda3/etc/profile.d/conda.sh
conda activate mvbodyfit

PY=/simurgh2/users/juze/anaconda3/envs/mvbodyfit/bin/python
SCRIPT=/simurgh/u/juze/code/mv-bodyfit/tools/precompute_vitpose_hoim3.py

# Mono v17/v19 are written to /simurgh2 NFS by the regen sbatch wrappers.
MONO_ROOT=/simurgh2/datasets/HOI-M3/mhr_mono
MASK_SHARD_ROOT=/simurgh2/datasets/HOI-M3/mask_shards
IMAGE_ROOT=/simurgh2/datasets/HOI-M3/images
OUTPUT_ROOT=/scr/juze/datasets/HOI-M3/vitpose_new6

if [ ! -d "$MONO_ROOT/$SEQ" ]; then
    echo "ERROR: $MONO_ROOT/$SEQ missing"
    exit 2
fi

cd /simurgh/u/juze/code/mv-bodyfit

t0=$(date +%s)
CUDA_VISIBLE_DEVICES=0 $PY $SCRIPT \
    --sequence "$SEQ" \
    --views 17 19 \
    --mono_root "$MONO_ROOT" \
    --mask_shard_root "$MASK_SHARD_ROOT" \
    --image_root "$IMAGE_ROOT" \
    --output_root "$OUTPUT_ROOT"
rc=$?
dt=$(($(date +%s) - t0))
echo "$(date) [$SEQ] vitpose v17,v19 finished rc=$rc in ${dt}s"
exit $rc
