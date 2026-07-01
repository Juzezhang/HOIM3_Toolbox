#!/bin/bash
#SBATCH --job-name=reconvert_shards
#SBATCH --partition=simurgh
#SBATCH --account=simurgh
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/simurgh/u/juze/regen_logs/reconvert_%A.out
#SBATCH --error=/simurgh/u/juze/regen_logs/reconvert_%A.err
#SBATCH --exclude=simurgh2

# Re-convert mask_npz_generated -> mask_shards for one sequence.
# Used to fix seqs whose old shards had zero data at views 17-24 despite
# the source npz having full 42-view content.
# Usage: sbatch sbatch_reconvert_shards.sh <seq>

set -u
SEQ=${1:?seq required}
echo "=== Job $SLURM_JOB_ID on $SLURMD_NODENAME — re-convert $SEQ ==="

source /simurgh2/users/juze/anaconda3/etc/profile.d/conda.sh
conda activate HOIM3_Toolbox
PY=/simurgh2/users/juze/anaconda3/envs/HOIM3_Toolbox/bin/python

TMP=/scr/juze/reconvert_tmp/$SEQ
DST=/simurgh2/datasets/HOI-M3
SRC=/simurgh2/datasets/HOI-M3/mask_npz_generated/$SEQ

mkdir -p "$TMP/mask_npz"
/usr/bin/rm -f "$TMP/mask_npz/$SEQ"
ln -sf "$SRC" "$TMP/mask_npz/$SEQ"

# Wipe old (broken) shards so converter rewrites cleanly
SHARD_DIR=$DST/mask_shards/$SEQ
if [ -d "$SHARD_DIR" ]; then
    echo "Wiping old shards: $SHARD_DIR"
    /usr/bin/rm -rf "$SHARD_DIR"
fi

t0=$(date +%s)
$PY /simurgh/u/juze/code/HOIM3_Toolbox/scripts/convert_masks_npz_to_lz4.py \
    --src_root "$TMP" \
    --dst_root "$DST" \
    --sequences "$SEQ" \
    --num_workers 10 \
    --compression_level 6
rc=$?
dt=$(($(date +%s) - t0))

if [ "$rc" -eq 0 ]; then
    /usr/bin/touch "$SHARD_DIR/.merged_1080p_done"
    /usr/bin/touch "$SHARD_DIR/.restored_swap24"
    echo "$(date) [$SEQ] reconvert OK in ${dt}s"
else
    echo "$(date) [$SEQ] reconvert FAIL rc=$rc in ${dt}s"
fi
exit $rc
