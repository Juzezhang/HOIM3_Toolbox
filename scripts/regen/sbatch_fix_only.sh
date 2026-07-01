#!/bin/bash
#SBATCH --job-name=mask_fix
#SBATCH --partition=simurgh
#SBATCH --account=simurgh
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=21-00:00:00
#SBATCH --output=/simurgh/u/juze/regen_logs/fix_%A.out
#SBATCH --error=/simurgh/u/juze/regen_logs/fix_%A.err
#SBATCH --exclude=simurgh2

set -u
echo "=== Job $SLURM_JOB_ID on $SLURMD_NODENAME ==="
SEQ=${1:?seq name required}
FIX=/simurgh/u/juze/code/HOIM3_Toolbox/scripts/regen/fix_one_seq_v2.sh

echo "$(date) [$SEQ] sbatch fix start"
bash $FIX $SEQ
rc=$?
echo "$(date) [$SEQ] sbatch fix finish rc=$rc"
exit $rc
