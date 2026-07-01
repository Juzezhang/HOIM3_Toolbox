#!/bin/bash
# Dispatcher for restore_swap24: throttle to MAX_QUEUE concurrent jobs.
#
# Usage:
#   bash dispatch_restore_swap24.sh                    # submit all 24 seqs
#   bash dispatch_restore_swap24.sh seq1 seq2 ...      # submit listed seqs only
#
# Env:
#   MAX_QUEUE  (default 4)
#   POLL_SEC   (default 30)
set -u

MAX_QUEUE=${MAX_QUEUE:-4}
POLL_SEC=${POLL_SEC:-30}
LOG_ROOT=/simurgh/u/juze/regen_logs
SBATCH_SCRIPT=/simurgh/u/juze/code/HOIM3_Toolbox/scripts/regen/sbatch_restore_swap24.sh

mkdir -p "$LOG_ROOT"

# Default seq list = all 24
DEFAULT_SEQS=(
    # Path A (16 — backup available)
    bedroom_data05 bedroom_data35
    diningroom_data09 diningroom_data11
    livingroom_data48
    office_data24 office_data28 office_data32 office_data33 office_data34
    office_data35 office_data36 office_data40 office_data41 office_data42
    office_data55
    # Path B (8 — inverse perm)
    livingroom_data03 livingroom_data04 livingroom_data30 livingroom_data31
    livingroom_data32
    office_data18 office_data30 office_data61
)

if [[ $# -gt 0 ]]; then
    SEQS=("$@")
else
    SEQS=("${DEFAULT_SEQS[@]}")
fi

echo "[dispatch] $(date) MAX_QUEUE=$MAX_QUEUE POLL_SEC=$POLL_SEC seqs=${#SEQS[@]}"

count_my_running() {
    squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -c '^restore_' || true
}

for SEQ in "${SEQS[@]}"; do
    while :; do
        n=$(count_my_running)
        if [[ "$n" -lt "$MAX_QUEUE" ]]; then
            break
        fi
        echo "[dispatch] $(date) queue=$n/$MAX_QUEUE — waiting"
        sleep "$POLL_SEC"
    done
    echo "[dispatch] $(date) submit restore_$SEQ"
    sbatch -J "restore_$SEQ" "$SBATCH_SCRIPT" "$SEQ" | tee -a "$LOG_ROOT/dispatch_restore_swap24.log"
    sleep 2
done

echo "[dispatch] $(date) all submissions done; waiting for completion"
while :; do
    n=$(count_my_running)
    if [[ "$n" -eq 0 ]]; then
        break
    fi
    echo "[dispatch] $(date) queue=$n still running"
    sleep "$POLL_SEC"
done
echo "[dispatch] $(date) all restore jobs complete"
