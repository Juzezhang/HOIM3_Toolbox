#!/bin/bash
#SBATCH --job-name=agg_convert
#SBATCH --partition=simurgh
#SBATCH --account=simurgh
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=/simurgh/u/juze/regen_logs/agg_convert_%A.out
#SBATCH --error=/simurgh/u/juze/regen_logs/agg_convert_%A.err
# simurgh5 overloaded (4 cutie recovery GPUs reading NFS heavily); exclude it for agg jobs
#SBATCH --exclude=simurgh2,simurgh5

# Aggregate (Cutie tracking → mask_npz_cutie) + Convert (mask_npz_cutie → mask_shards)
# for one seq. Self-contained: uses only NFS paths, runs on any simurgh node.
#
# Usage:  sbatch sbatch_cutie_agg_convert.sh <seq>

set -u
SEQ=${1:?seq required}
echo "=== Job $SLURM_JOB_ID on $SLURMD_NODENAME — agg+convert $SEQ ==="

PY=/simurgh2/users/juze/anaconda3/envs/HOIM3_Toolbox/bin/python
TRACK_ROOT=/simurgh2/datasets/HOI-M3/cutie_tracking
AGG_ROOT=/simurgh2/datasets/HOI-M3/mask_npz_cutie
SHARD_ROOT=/simurgh2/datasets/HOI-M3/mask_shards
# Use /tmp on the compute node for convert-script symlink trick
TMP_DIR="/tmp/cutie_agg_${SEQ}_${SLURM_JOB_ID}"
mkdir -p "$AGG_ROOT" "$TMP_DIR"

if [ -f "$SHARD_ROOT/$SEQ/.restored_swap24" ]; then
    echo "Sentinel already exists, exit"; exit 0
fi

# Step 3: aggregate (idempotent — skips frames already in $AGG_ROOT/$SEQ/)
echo "$(date) [$SEQ] aggregate START"
$PY /simurgh/u/juze/code/HOIM3_Toolbox/scripts/regen/aggregate_cutie_to_npz.py \
    --seq "$SEQ" \
    --cutie_root "$TRACK_ROOT" \
    --output_root "$AGG_ROOT" \
    --workers 12
rc=$?
if [ "$rc" -ne 0 ]; then
    echo "$(date) [$SEQ] aggregate FAIL rc=$rc"; exit $rc
fi
echo "$(date) [$SEQ] aggregate OK"

# Step 4: convert via tmp symlink (script expects src_root/mask_npz/<seq>/)
mkdir -p "$TMP_DIR/mask_npz"
ln -sf "$AGG_ROOT/$SEQ" "$TMP_DIR/mask_npz/$SEQ"

echo "$(date) [$SEQ] convert START"
$PY /simurgh/u/juze/code/HOIM3_Toolbox/scripts/convert_masks_npz_to_lz4.py \
    --src_root "$TMP_DIR" \
    --dst_root /simurgh2/datasets/HOI-M3 \
    --sequences "$SEQ" \
    --num_workers 12 \
    --compression_level 6
rc=$?
if [ "$rc" -ne 0 ]; then
    echo "$(date) [$SEQ] convert FAIL rc=$rc"
    /usr/bin/rm -rf "$TMP_DIR"
    exit $rc
fi

/usr/bin/touch "$SHARD_ROOT/$SEQ/.merged_1080p_done"
/usr/bin/touch "$SHARD_ROOT/$SEQ/.restored_swap24"
echo "$(date) [$SEQ] convert OK — sentinels written"

/usr/bin/rm -rf "$TMP_DIR"
echo "$(date) [$SEQ] DONE"
exit 0
