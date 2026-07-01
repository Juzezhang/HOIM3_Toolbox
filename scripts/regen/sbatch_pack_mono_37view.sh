#!/bin/bash
#SBATCH --job-name=pack37
#SBATCH --partition=simurgh
#SBATCH --account=simurgh
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/simurgh/u/juze/regen_logs/pack37_%A.out
#SBATCH --error=/simurgh/u/juze/regen_logs/pack37_%A.err

# Pack one HOI-M3 seq into the 37-view packed cache.
# CPU-only; reads NFS mono, writes NFS packed.
# Usage: sbatch sbatch_pack_mono_37view.sh <sequence>

set -u
SEQ=${1:?seq required}
echo "=== Job $SLURM_JOB_ID on $SLURMD_NODENAME for $SEQ ==="

PY=/simurgh2/users/juze/anaconda3/envs/mvbodyfit/bin/python
SCRIPT=/simurgh/u/juze/code/HOIM3_Toolbox/scripts/regen/pack_mono_cache.py

MONO_ROOT=/simurgh2/datasets/HOI-M3/mhr_mono
OUTPUT_ROOT=/simurgh2/datasets/HOI-M3/mhr_mono_packed

# Skip if pack already at 37 views.
META="$OUTPUT_ROOT/$SEQ/person0/meta.npz"
if [ -f "$META" ]; then
    nv=$($PY -c "import numpy as np; m=np.load('$META'); print(int(m['views'].shape[0]))" 2>/dev/null)
    if [ "$nv" = "37" ]; then
        echo "[$SEQ] already packed at 37 views; exiting"
        exit 0
    fi
    echo "[$SEQ] removing stale ${nv}-view pack"
    rm -rf "$OUTPUT_ROOT/$SEQ"
fi

mkdir -p "$OUTPUT_ROOT"

t0=$(date +%s)
$PY $SCRIPT \
    --sequence "$SEQ" \
    --mono_root "$MONO_ROOT" \
    --output_root "$OUTPUT_ROOT" \
    --views 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 34 35 36 37 38 39 41 \
    --workers 16
rc=$?
dt=$(($(date +%s) - t0))
echo "$(date) [$SEQ] pack37 finished rc=$rc in ${dt}s"

# Verify 37 views before declaring success.
if [ $rc -eq 0 ] && [ -f "$META" ]; then
    nv=$($PY -c "import numpy as np; m=np.load('$META'); print(int(m['views'].shape[0]))" 2>/dev/null)
    if [ "$nv" != "37" ]; then
        echo "[$SEQ] post-pack verification FAILED: views=$nv"
        exit 2
    fi
fi
exit $rc
