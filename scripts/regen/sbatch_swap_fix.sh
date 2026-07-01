#!/bin/bash
#SBATCH --job-name=swap_fix
#SBATCH --partition=simurgh
#SBATCH --account=simurgh
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/simurgh/u/juze/regen_logs/swap_fix_%A.out
#SBATCH --error=/simurgh/u/juze/regen_logs/swap_fix_%A.err
#SBATCH --exclude=simurgh2

set -u
SEQ=${1:?seq required}
echo "=== Job $SLURM_JOB_ID on $SLURMD_NODENAME for $SEQ ==="
PY=/simurgh2/users/juze/anaconda3/envs/HOIM3_Toolbox/bin/python
SCRIPT=/scr/juze/fix_person_swap_v3.py
FIX=/simurgh/u/juze/code/HOIM3_Toolbox/scripts/regen/fix_one_seq_v2.sh
LOG_DIR=/scr/juze/swap24_logs
DONE_DIR=/scr/juze/swap24_done
NPZ_DIR=/simurgh2/datasets/HOI-M3/mask_npz_generated/$SEQ
LOG=$LOG_DIR/${SEQ}.log

mkdir -p $LOG_DIR $DONE_DIR

[ -f $DONE_DIR/${SEQ}.done ] && { echo "$(date) [$SEQ] already done" >> $LOG; exit 0; }
[ ! -d "$NPZ_DIR" ] && { echo "$(date) [$SEQ] no NPZ_DIR" >> $LOG; exit 1; }

# Prefer shared NFS copy of the fix script (works on non-simurgh5 nodes)
if [ -f /simurgh/u/juze/code/HOIM3_Toolbox/scripts/regen/fix_person_swap_v3.py ]; then
    SCRIPT=/simurgh/u/juze/code/HOIM3_Toolbox/scripts/regen/fix_person_swap_v3.py
fi

echo "$(date) [$SEQ] sbatch FIX START on $SLURMD_NODENAME (job $SLURM_JOB_ID)" >> $LOG

$PY $SCRIPT --npz_dir $NPZ_DIR --pass1_workers 6 --pass2_workers 6 >> $LOG 2>&1
rc=$?
if [ $rc -ne 0 ]; then
    echo "$(date) [$SEQ] fix script FAILED rc=$rc" >> $LOG
    exit $rc
fi

echo "$(date) [$SEQ] running converter..." >> $LOG
bash $FIX $SEQ >> $LOG 2>&1
rc2=$?
if [ $rc2 -eq 0 ]; then
    touch $DONE_DIR/${SEQ}.done
    echo "$(date) [$SEQ] sbatch FIX DONE" >> $LOG
    exit 0
else
    echo "$(date) [$SEQ] convert FAILED rc=$rc2" >> $LOG
    exit $rc2
fi
