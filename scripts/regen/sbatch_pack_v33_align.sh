#!/bin/bash
#SBATCH --job-name=pack_v33_align
#SBATCH --partition=simurgh
#SBATCH --account=simurgh
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --exclude=simurgh2
#SBATCH --output=/simurgh/u/juze/regen_logs/pack_v33_align_%A.out
#SBATCH --error=/simurgh/u/juze/regen_logs/pack_v33_align_%A.err

# Pack v33 alignment data: mono v33 NPZ + fit keypoints3d JSON → packed .npy
# CPU only, no GPU needed.
# Usage: sbatch sbatch_pack_v33_align.sh <SEQ>

set -u
SEQ=${1:?seq required}
echo "=== Job $SLURM_JOB_ID on $SLURMD_NODENAME for $SEQ ==="

PY=/simurgh2/users/juze/anaconda3/envs/mvbodyfit/bin/python
SCRIPT=/scr/juze/swap24_cleanup_logs/pack_v33_align_data.py

t0=$(date +%s)
$PY $SCRIPT --sequence "$SEQ" --workers 4
rc=$?
dt=$(($(date +%s) - t0))
echo "$(date) [$SEQ] pack_v33_align finished rc=$rc in ${dt}s"
exit $rc
