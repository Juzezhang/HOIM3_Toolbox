#!/bin/bash
# Top-level dispatcher for Cutie mask-tracking pipeline (Step 2 + 3 + 4).
# Builds pending list (11 seqs × 42 views = 462 jobs), throttles SLURM
# submissions, and after all 42 views of a seq finish runs the aggregator
# (Step 3) + LZ4 shard converter (Step 4) and writes .restored_swap24 sentinel.
#
# Local simurgh5 GPU workers (4 GPUs) are launched separately:
#     bash cutie_track_local_worker.sh 0 &
#     bash cutie_track_local_worker.sh 1 &
#     bash cutie_track_local_worker.sh 2 &
#     bash cutie_track_local_worker.sh 3 &
# Both this dispatcher and the local workers consume from the same
# PENDING_FILE atomically (mkdir lock + grep-out).
#
# Run in background:
#     nohup bash dispatch_cutie_tracking.sh </dev/null \
#         >/scr/juze/swap24_cleanup_logs/cutie_dispatcher.log 2>&1 &

set -u

SEQS=(
    office_data24 office_data28 office_data32 office_data33 office_data34
    office_data35 office_data36 office_data40 office_data41 office_data42
    office_data55
)
N_VIEWS=42

PENDING_FILE=/scr/juze/cutie_tracking_pending.txt
DONE_FILE=/scr/juze/cutie_tracking_done.list
LOCK_DIR=/scr/juze/cutie_track_locks
LOG=/scr/juze/swap24_cleanup_logs/cutie_dispatcher.log
SBATCH_SCRIPT=/simurgh/u/juze/code/HOIM3_Toolbox/scripts/regen/sbatch_cutie_track.sh
OUTPUT_ROOT=/simurgh2/datasets/HOI-M3/cutie_tracking
REF_ROOT=/simurgh2/datasets/HOI-M3/cutie_refs
AGG_ROOT=/simurgh2/datasets/HOI-M3/mask_npz_cutie
SHARD_ROOT=/simurgh2/datasets/HOI-M3/mask_shards
TMP_CONVERT_ROOT=/scr/juze/cutie_convert_tmp
MAX_SBATCH_QUEUE=24
POLL_SECS=30
CONDA_BASE=/simurgh2/users/juze/anaconda3

mkdir -p "$LOCK_DIR" "$(dirname $LOG)" "$AGG_ROOT" "$TMP_CONVERT_ROOT"
touch "$DONE_FILE"

set +u
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"
set -u

log() { echo "$(date '+%F %T') [dispatch] $*"; }

build_pending() {
    : > "$PENDING_FILE"
    local total=0
    for seq in "${SEQS[@]}"; do
        for v in $(seq 0 $((N_VIEWS-1))); do
            sentinel="$OUTPUT_ROOT/$seq/$v/.tracked_done"
            if [ -f "$sentinel" ]; then continue; fi
            echo "$seq $v" >> "$PENDING_FILE"
            total=$((total+1))
        done
    done
    log "pending built: $total view-jobs in $PENDING_FILE"
}

count_sbatch_queue() {
    squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -c "^cutie_trk" || true
}

remove_from_pending() {
    local seq=$1 view=$2
    grep -vxF "$seq $view" "$PENDING_FILE" > "${PENDING_FILE}.tmp$$" 2>/dev/null || true
    mv "${PENDING_FILE}.tmp$$" "$PENDING_FILE" 2>/dev/null || true
}

submit_one() {
    local seq=$1 view=$2
    local lock="$LOCK_DIR/${seq}_${view}.lock"
    if ! mkdir "$lock" 2>/dev/null; then
        # Already claimed (in-flight or local worker took it); just remove from pending.
        remove_from_pending "$seq" "$view"
        return 2
    fi
    remove_from_pending "$seq" "$view"
    local jid
    jid=$(sbatch "$SBATCH_SCRIPT" "$seq" "$view" 2>&1 | awk '{print $NF}')
    log "submit sbatch seq=$seq v=$view -> jobid=$jid"
    # Keep the lock dir as a "submitted" marker; it's idempotent if dispatcher restarts.
    return 0
}

aggregate_and_convert_seq() {
    local seq=$1
    log "POST seq=$seq starting aggregate + convert"
    # Step 3 aggregate (HOIM3_Toolbox env: needs np, no torch)
    set +u
    conda activate HOIM3_Toolbox
    set -u
    python /simurgh/u/juze/code/HOIM3_Toolbox/scripts/regen/aggregate_cutie_to_npz.py \
        --seq "$seq" \
        --cutie_root "$OUTPUT_ROOT" \
        --output_root "$AGG_ROOT" \
        --workers 8 \
        >> "/scr/juze/swap24_cleanup_logs/cutie_agg_${seq}.log" 2>&1
    local rc=$?
    if [ $rc -ne 0 ]; then
        log "POST seq=$seq aggregate FAIL rc=$rc"
        return 1
    fi
    # Step 4: convert to LZ4 shards. Converter requires <src_root>/mask_npz/<seq>.
    # Build a temp src_root symlink.
    local tmp_src="$TMP_CONVERT_ROOT/$seq"
    mkdir -p "$tmp_src/mask_npz"
    ln -sfn "$AGG_ROOT/$seq" "$tmp_src/mask_npz/$seq"
    rm -rf "$SHARD_ROOT/$seq"
    python /simurgh/u/juze/code/HOIM3_Toolbox/scripts/convert_masks_npz_to_lz4.py \
        --src_root "$tmp_src" \
        --dst_root /simurgh2/datasets/HOI-M3 \
        --sequences "$seq" \
        --compression_level 6 \
        --num_workers 12 \
        >> "/scr/juze/swap24_cleanup_logs/cutie_convert_${seq}.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        log "POST seq=$seq convert FAIL rc=$rc"
        return 1
    fi
    touch "$SHARD_ROOT/$seq/.merged_1080p_done"
    touch "$SHARD_ROOT/$seq/.restored_swap24"
    log "POST seq=$seq OK (shards at $SHARD_ROOT/$seq)"
    set +u
    conda deactivate
    set -u
}

# 1. Build pending
build_pending

# 2. Submission loop (keep MAX_SBATCH_QUEUE sbatch jobs in flight)
while [ -s "$PENDING_FILE" ]; do
    qcount=$(count_sbatch_queue)
    if [ "$qcount" -lt "$MAX_SBATCH_QUEUE" ]; then
        line=$(head -1 "$PENDING_FILE" 2>/dev/null || true)
        if [ -z "$line" ]; then sleep "$POLL_SECS"; continue; fi
        seq=$(echo "$line" | awk '{print $1}')
        view=$(echo "$line" | awk '{print $2}')
        if [ -z "$seq" ] || [ -z "$view" ]; then
            sed -i '1d' "$PENDING_FILE" 2>/dev/null || true
            continue
        fi
        # If sentinel already exists (local worker beat us), skip
        if [ -f "$OUTPUT_ROOT/$seq/$view/.tracked_done" ]; then
            grep -vxF "$seq $view" "$PENDING_FILE" > "${PENDING_FILE}.tmp$$" 2>/dev/null || true
            mv "${PENDING_FILE}.tmp$$" "$PENDING_FILE" 2>/dev/null || true
            continue
        fi
        submit_one "$seq" "$view" || sleep 2
    else
        sleep "$POLL_SECS"
    fi
done

log "all submissions done — waiting for sentinels"

# 3. Wait until every (seq, view) sentinel exists, then post-process per seq.
done_seqs_file=/scr/juze/cutie_tracking_post_done.list
touch "$done_seqs_file"

while true; do
    all_done=true
    for seq in "${SEQS[@]}"; do
        if grep -qxF "$seq" "$done_seqs_file" 2>/dev/null; then continue; fi
        complete=true
        for v in $(seq 0 $((N_VIEWS-1))); do
            if [ ! -f "$OUTPUT_ROOT/$seq/$v/.tracked_done" ]; then
                complete=false; break
            fi
        done
        if $complete; then
            log "seq=$seq all $N_VIEWS views tracked — running post"
            if aggregate_and_convert_seq "$seq"; then
                echo "$seq" >> "$done_seqs_file"
            else
                log "seq=$seq post FAILED — leaving to retry next loop"
                all_done=false
            fi
        else
            all_done=false
        fi
    done
    if $all_done; then break; fi
    sleep 60
done

log "ALL DONE — 11 seqs converted to shards with .restored_swap24"
