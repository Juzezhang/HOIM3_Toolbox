#!/bin/bash
#SBATCH --job-name=mono_lo
#SBATCH --partition=sc-loprio
#SBATCH --account=default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=/simurgh/u/juze/regen_logs/mono_lo_%j.out
#SBATCH --error=/simurgh/u/juze/regen_logs/mono_lo_%j.err
#SBATCH --requeue

# Run SAM-3D-Body mono on sc-loprio (preemptable, but plenty of capacity).
# Usage: sbatch sbatch_mono_loprio.sh <seq> <view>

set -u
SEQ=${1:?seq required}
VIEW=${2:?view required}
echo "=== Job $SLURM_JOB_ID on $SLURMD_NODENAME for $SEQ v$VIEW ==="
nvidia-smi -L

source /simurgh2/users/juze/anaconda3/etc/profile.d/conda.sh
conda activate fastsam3d

PY=/simurgh2/users/juze/anaconda3/envs/fastsam3d/bin/python
SCRIPT=/simurgh/u/juze/code/fast-sam-3d-body/tools/process_hoim3.py
CKPT=/simurgh/u/juze/code/fast-sam-3d-body/checkpoints/sam-3d-body-dinov3/model.ckpt
MHR=/simurgh/u/juze/code/fast-sam-3d-body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt

ROOT=/simurgh2/datasets/HOI-M3
SHARD_ROOT=/simurgh2/datasets/HOI-M3/mask_shards
OUTPUT_ROOT=/simurgh2/datasets/HOI-M3/mhr_mono
VAL=/tmp/nonexistent_validity_dir_xyz

/usr/bin/rm -f $OUTPUT_ROOT/$SEQ/.done_step1_body
mkdir -p $OUTPUT_ROOT

t0=$(date +%s)
$PY $SCRIPT --checkpoint_path $CKPT --mhr_path $MHR \
    --inference_type body --views "$VIEW" --root_path "$ROOT" \
    --mask_shard_root "$SHARD_ROOT" --validity_root "$VAL" \
    --output_root "$OUTPUT_ROOT" \
    --ignore_seq_marker \
    --sequences "$SEQ"
rc=$?
dt=$(($(date +%s) - t0))
echo "$(date) [$SEQ] mono v$VIEW loprio rc=$rc in ${dt}s"
exit $rc
