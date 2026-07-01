#!/bin/bash
#SBATCH --job-name=mono_v17
#SBATCH --partition=simurgh
#SBATCH --account=simurgh
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=21-00:00:00
#SBATCH --output=/simurgh/u/juze/regen_logs/mono_v17_%A.out
#SBATCH --error=/simurgh/u/juze/regen_logs/mono_v17_%A.err
#SBATCH --exclude=simurgh2

# Run SAM-3D-Body mono MHR inference on HOI-M3 view 17 for one sequence.
# Outputs to node-local /scr/$USER (any simurgh node has it as local SSD) if writable,
# else fall back to /simurgh2 NFS. Watcher / final rsync mirrors back to NFS.
# Usage: sbatch sbatch_mono_view17.sh <sequence_name>

set -u
SEQ=${1:?seq required}
echo "=== Job $SLURM_JOB_ID on $SLURMD_NODENAME for $SEQ ==="
nvidia-smi -L

source /simurgh2/users/juze/anaconda3/etc/profile.d/conda.sh
conda activate fastsam3d

PY=/simurgh2/users/juze/anaconda3/envs/fastsam3d/bin/python
SCRIPT=/simurgh/u/juze/code/fast-sam-3d-body/tools/process_hoim3.py
CKPT=/simurgh/u/juze/code/fast-sam-3d-body/checkpoints/sam-3d-body-dinov3/model.ckpt
MHR=/simurgh/u/juze/code/fast-sam-3d-body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt

ROOT=/simurgh2/datasets/HOI-M3
SHARD_ROOT=/simurgh2/datasets/HOI-M3/mask_shards
NFS_MIRROR=/simurgh2/datasets/HOI-M3/mhr_mono        # final NFS dest

# Pick output location:
# - If NFS already has substantial existing npz (≥1000 in person0/) → use NFS directly
#   (skip slow seed rsync; accept NFS metadata cost for skip-existing checks)
# - Else if /scr writable → use /scr (fresh seq, fast)
# - Else fall back to NFS
LOCAL_SCR=/scr/juze/datasets/HOI-M3/mhr_mono
EXISTING_NPZ=$(/usr/bin/find "$NFS_MIRROR/$SEQ/17/person0" -maxdepth 1 -name '*.npz' 2>/dev/null | head -1001 | wc -l)
if [ "$EXISTING_NPZ" -ge 1000 ]; then
  OUTPUT_ROOT=$NFS_MIRROR
  USE_LOCAL=0
  echo "[output] NFS already has $EXISTING_NPZ npz for v17 — skipping seed, using NFS directly"
elif mkdir -p "$LOCAL_SCR" 2>/dev/null && [ -w /scr/juze ]; then
  OUTPUT_ROOT=$LOCAL_SCR
  USE_LOCAL=1
  echo "[output] using LOCAL $OUTPUT_ROOT (will rsync to NFS at end)"
else
  OUTPUT_ROOT=$NFS_MIRROR
  USE_LOCAL=0
  echo "[output] /scr not writable on $SLURMD_NODENAME — using NFS $OUTPUT_ROOT directly"
fi
VAL=/tmp/nonexistent_validity_dir_xyz   # bypass per-frame validity (script falls back to all-valid)

# ── Cross-cluster coordination: atomic-claim via GCS before running ──
VIEW=17
GCS_BASE=gs://data-storage-0/HOI-M3/mhr_mono
SENTINEL_GCS="$GCS_BASE/$SEQ/$VIEW/.view_done_$VIEW"
CLAIM_GCS="$GCS_BASE/$SEQ/$VIEW/.claim"
GSUTIL=/simurgh2/users/juze/anaconda3/envs/SAMPA/bin/gsutil

# Skip if another cluster already finished
if BOTO_CONFIG=/simurgh/u/juze/.boto_gcs $GSUTIL -q ls "$SENTINEL_GCS" >/dev/null 2>&1; then
  echo "[claim] view already done on GCS: $SENTINEL_GCS — skipping"
  exit 0
fi

# Atomic claim (fails if .claim already exists)
echo "[claim] attempting atomic claim at $CLAIM_GCS"
claim_out=$(echo "$SLURMD_NODENAME sbatch=$SLURM_JOB_ID $(date -Is)" | \
  BOTO_CONFIG=/simurgh/u/juze/.boto_gcs $GSUTIL -q -h "x-goog-if-generation-match:0" \
    cp - "$CLAIM_GCS" 2>&1)
claim_rc=$?
if [ $claim_rc -ne 0 ]; then
  if echo "$claim_out" | grep -qiE 'PreconditionFailed|412|already exists'; then
    echo "[claim] TAKEN by another worker — exiting cleanly"
    exit 0
  fi
  echo "[claim] ERR rc=$claim_rc out=$claim_out — proceeding anyway (best-effort)"
fi
echo "[claim] CLAIMED $SEQ v$VIEW"

# Seed local /scr with existing NFS outputs for THIS VIEW ONLY (not whole seq —
# that would copy all other views' npz, can be millions of files / hours).
if [ "$USE_LOCAL" = "1" ] && [ -d "$NFS_MIRROR/$SEQ/$VIEW" ]; then
  echo "[seed] rsync existing $NFS_MIRROR/$SEQ/$VIEW → $OUTPUT_ROOT/$SEQ/$VIEW"
  mkdir -p "$OUTPUT_ROOT/$SEQ"
  rsync -a --info=stats1 "$NFS_MIRROR/$SEQ/$VIEW/" "$OUTPUT_ROOT/$SEQ/$VIEW/" || true
fi

# Remove stale completion marker for THIS view+inference combo
rm -f "$OUTPUT_ROOT/$SEQ/.done_step1_body" "$NFS_MIRROR/$SEQ/.done_step1_body"

t0=$(date +%s)
$PY $SCRIPT --checkpoint_path $CKPT --mhr_path $MHR \
    --inference_type body --views "17" --root_path "$ROOT" \
    --mask_shard_root "$SHARD_ROOT" --validity_root "$VAL" \
    --output_root "$OUTPUT_ROOT" \
    --ignore_seq_marker \
    --sequences "$SEQ"
rc=$?
dt=$(($(date +%s) - t0))
echo "$(date) [$SEQ] mono v17 finished rc=$rc in ${dt}s"

# IMMEDIATE: write completion sentinel + remove claim BEFORE the npz rsync.
# Semantics: .view_done means "inference complete, npz upload may still be async".
# This closes the race window where another worker could redo the (seq, view) while
# we're still rsync'ing /scr → NFS (1-2 min). Cross-cluster coordination prioritizes
# correctness (no duplicate inference) over npz-immediate-availability.
if [ $rc -eq 0 ]; then
  SENTINEL="$NFS_MIRROR/$SEQ/$VIEW/.view_done_$VIEW"
  mkdir -p "$NFS_MIRROR/$SEQ/$VIEW"
  echo "$SLURMD_NODENAME $(date -Is) sbatch=$SLURM_JOB_ID inference_complete" > "$SENTINEL"
  BOTO_CONFIG=/simurgh/u/juze/.boto_gcs $GSUTIL cp "$SENTINEL" "$SENTINEL_GCS" 2>&1 \
    | tail -1 | sed "s|^|[gcs-sentinel] |"
  BOTO_CONFIG=/simurgh/u/juze/.boto_gcs $GSUTIL -q rm "$CLAIM_GCS" 2>/dev/null \
    && echo "[claim] released $CLAIM_GCS"
  echo "[view-done] sentinel + claim release: race window closed"
fi

# AFTER sentinel: now sync local /scr → NFS (slower, ~1-2 min, but no race risk).
if [ $rc -eq 0 ] && [ "$USE_LOCAL" = "1" ]; then
  echo "[final-sync] rsync $OUTPUT_ROOT/$SEQ → $NFS_MIRROR/$SEQ (async, can take ~1-2 min)"
  mkdir -p "$NFS_MIRROR"
  rsync -a "$OUTPUT_ROOT/$SEQ/$VIEW/" "$NFS_MIRROR/$SEQ/$VIEW/" && echo "[final-sync] OK"
fi
exit $rc
