#!/bin/bash
#SBATCH --job-name=pack_mono
#SBATCH --partition=simurgh
#SBATCH --account=simurgh
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --exclude=simurgh2
#SBATCH --output=/simurgh/u/juze/regen_logs/pack_mono_%A.out
#SBATCH --error=/simurgh/u/juze/regen_logs/pack_mono_%A.err

# Pack per-frame HOI-M3 mono MHR NPZs into a packed cache for fast bulk-load.
# Reads NFS mono, writes NFS packed — runs on any simurgh node.
# Usage: sbatch sbatch_pack_mono.sh <sequence_name>

set -u
SEQ=${1:?seq required}
echo "=== Job $SLURM_JOB_ID on $SLURMD_NODENAME for $SEQ ==="

PY=/simurgh2/users/juze/anaconda3/envs/mvbodyfit/bin/python
SCRIPT=/simurgh/u/juze/code/HOIM3_Toolbox/scripts/regen/pack_mono_cache.py

MONO_ROOT=/simurgh2/datasets/HOI-M3/mhr_mono
OUTPUT_ROOT=/simurgh2/datasets/HOI-M3/mhr_mono_packed

mkdir -p "$OUTPUT_ROOT"

t0=$(date +%s)
$PY $SCRIPT \
    --sequence "$SEQ" \
    --mono_root "$MONO_ROOT" \
    --output_root "$OUTPUT_ROOT" \
    --views 0 2 5 6 7 8 10 11 14 15 17 19 21 22 23 24 25 26 27 28 29 \
    --workers 8
rc=$?
dt=$(($(date +%s) - t0))
echo "$(date) [$SEQ] pack_mono finished rc=$rc in ${dt}s"
exit $rc
