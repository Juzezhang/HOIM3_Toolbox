#!/bin/bash
#SBATCH --job-name=restore_swap24
#SBATCH --partition=simurgh
#SBATCH --account=simurgh
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --exclude=simurgh2
#SBATCH --output=/simurgh/u/juze/regen_logs/restore_swap24_%A_%x.out
#SBATCH --error=/simurgh/u/juze/regen_logs/restore_swap24_%A_%x.err

# Usage:
#   sbatch -J restore_<seq> sbatch_restore_swap24.sh <seq>
#   (set job name via -J so logs include seq)
set -u
echo "=== restore_swap24 $SLURM_JOB_ID on $SLURMD_NODENAME ==="
SEQ=${1:?seq name required}
echo "[$SEQ] $(date) sbatch restore start"

bash /simurgh/u/juze/code/HOIM3_Toolbox/scripts/regen/restore_one_seq.sh "$SEQ"
rc=$?

echo "[$SEQ] $(date) sbatch restore finish rc=$rc"
exit $rc
