#!/bin/bash
#SBATCH --job-name=vitpose_new6
#SBATCH --partition=simurgh
#SBATCH --account=simurgh
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=21-00:00:00
#SBATCH --output=/simurgh/u/juze/regen_logs/vitpose_new6_%A.out
#SBATCH --error=/simurgh/u/juze/regen_logs/vitpose_new6_%A.err
#SBATCH --nodelist=simurgh5

# Run ViTPose precompute on HOI-M3 for the 6 NEW views (17,19,21,22,23,24).
# Output goes to a SEPARATE directory /scr/juze/datasets/HOI-M3/vitpose_new6
# (NOT touching the existing vitpose/ dir which has the legacy 10 views).
#
# Pinned to simurgh5 because mhr_mono for new views lives on /scr (simurgh5-local).
# Images and mask_shards are on /simurgh2 NFS.
#
# Per-seq: missing-mono views will produce all-zero slots; views array in meta.npz
# will still contain all 6 new-view names. After merge with legacy 10-view cache,
# the loader treats zero-conf rows as not-detected — equivalent to absent.
#
# Usage: sbatch sbatch_vitpose_new6.sh <sequence_name>

set -u
SEQ=${1:?seq required}
echo "=== Job $SLURM_JOB_ID on $SLURMD_NODENAME for $SEQ ==="
nvidia-smi -L

source /simurgh2/users/juze/anaconda3/etc/profile.d/conda.sh
conda activate mvbodyfit

PY=/simurgh2/users/juze/anaconda3/envs/mvbodyfit/bin/python
SCRIPT=/simurgh/u/juze/code/mv-bodyfit/tools/precompute_vitpose_hoim3.py

# Mono lives on /scr (local to simurgh5). Images + masks on /simurgh2 NFS.
MONO_ROOT=/scr/juze/datasets/HOI-M3/mhr_mono
MASK_SHARD_ROOT=/simurgh2/datasets/HOI-M3/mask_shards
IMAGE_ROOT=/simurgh2/datasets/HOI-M3/images
OUTPUT_ROOT=/scr/juze/datasets/HOI-M3/vitpose_new6

# Sanity: mono must exist for this seq.
if [ ! -d "$MONO_ROOT/$SEQ" ]; then
    echo "ERROR: $MONO_ROOT/$SEQ missing — no mono data to read bboxes from"
    exit 2
fi

# NOTE: NOT wiping output dir — incremental writes are fine for the new dir
# (we want re-runs to be idempotent and not destroy partial progress if a job
# is preempted).

cd /simurgh/u/juze/code/mv-bodyfit

t0=$(date +%s)
CUDA_VISIBLE_DEVICES=0 $PY $SCRIPT \
    --sequence "$SEQ" \
    --views 17 19 21 22 23 24 \
    --mono_root "$MONO_ROOT" \
    --mask_shard_root "$MASK_SHARD_ROOT" \
    --image_root "$IMAGE_ROOT" \
    --output_root "$OUTPUT_ROOT"
rc=$?
dt=$(($(date +%s) - t0))
echo "$(date) [$SEQ] vitpose_new6 finished rc=$rc in ${dt}s"
exit $rc
