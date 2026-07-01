#!/bin/bash
# Restore one HOI-M3 sequence's mask_shards to pre-swap_fix state.
#
# Auto-routes to Path A (re-convert from /simurgh/group/mask_npz backup) or
# Path B (inverse-perm on existing shard) based on presence of mask_npz backup.
#
# Usage:
#   bash restore_one_seq.sh <seq> [--force-path-b]
set -euo pipefail

SEQ=${1:?"sequence name required"}
FORCE_B="${2:-}"

PY=/simurgh2/users/juze/anaconda3/envs/HOIM3_Toolbox/bin/python
TOOLBOX=/simurgh/u/juze/code/HOIM3_Toolbox
CONVERT=$TOOLBOX/scripts/convert_masks_npz_to_lz4.py
UPGRADE=$TOOLBOX/scripts/regen/upgrade_to_1080p.py
RESTORE_B=$TOOLBOX/scripts/regen/restore_inverse_perm.py

GROUP_ROOT=/simurgh/group/juze/datasets/HOI-M3
SIMURGH2_ROOT=/simurgh2/datasets/HOI-M3
SHARD_ROOT=$SIMURGH2_ROOT/mask_shards
TMP_ROOT=$SIMURGH2_ROOT/mask_shards_restore_tmp
SCR_ROOT=/scr/juze/datasets/HOI-M3

NPZ_SRC=$GROUP_ROOT/mask_npz/$SEQ
SHARD_DIR=$SHARD_ROOT/$SEQ
BAK_DIR=$SHARD_ROOT/${SEQ}.swap_fix.bak

date
echo "=== restore_one_seq $SEQ ==="

if [[ ! -d "$SHARD_DIR" ]]; then
    echo "[FAIL] $SEQ: shard dir $SHARD_DIR missing"
    exit 2
fi

# Detect path
if [[ "$FORCE_B" == "--force-path-b" ]]; then
    USE_PATH=B
elif [[ -d "$NPZ_SRC" ]] && [[ -n "$(ls -A "$NPZ_SRC" 2>/dev/null | head -1)" ]]; then
    USE_PATH=A
else
    USE_PATH=B
fi
echo "[$SEQ] path=$USE_PATH"

# Safety: don't redo if already restored (look for .restored sentinel)
if [[ -f "$SHARD_DIR/.restored_swap24" ]]; then
    echo "[$SEQ] already restored (sentinel present) — skipping"
    exit 0
fi

if [[ "$USE_PATH" == "A" ]]; then
    # ---- PATH A ----
    # Step 1: convert NPZ → 720p shard in tmp dst
    TMP_SEQ_DST=$TMP_ROOT/$SEQ
    if [[ -d "$TMP_SEQ_DST/mask_shards/$SEQ" ]]; then
        echo "[$SEQ] tmp dst already exists at $TMP_SEQ_DST — removing for clean run"
        rm -rf "$TMP_SEQ_DST"
    fi
    mkdir -p "$TMP_SEQ_DST"
    echo "[$SEQ] PATH A step 1: convert mask_npz → 720p shard"
    $PY $CONVERT \
        --src_root "$GROUP_ROOT" \
        --dst_root "$TMP_SEQ_DST" \
        --sequences "$SEQ" \
        --num_workers 8 \
        --compression_level 6

    TMP_SHARD=$TMP_SEQ_DST/mask_shards/$SEQ
    if [[ ! -f "$TMP_SHARD/meta.json" ]]; then
        echo "[FAIL] $SEQ: convert produced no meta.json at $TMP_SHARD"
        exit 3
    fi

    # Step 2: move existing broken shard → .swap_fix.bak (keep for forensics)
    if [[ -d "$BAK_DIR" ]]; then
        echo "[$SEQ] .swap_fix.bak already exists — leaving previous bak intact, "
        echo "        moving current to .swap_fix.bak2"
        mv "$SHARD_DIR" "${BAK_DIR}2"
    else
        echo "[$SEQ] PATH A step 2: mv $SHARD_DIR → $BAK_DIR"
        mv "$SHARD_DIR" "$BAK_DIR"
    fi

    # Step 3: move new 720p shard into place
    echo "[$SEQ] PATH A step 3: install new 720p shard"
    mv "$TMP_SHARD" "$SHARD_DIR"

    # Cleanup empty tmp parent
    rmdir "$TMP_SEQ_DST/mask_shards" 2>/dev/null || true
    rmdir "$TMP_SEQ_DST" 2>/dev/null || true

    # Step 4: upgrade to 1080p
    echo "[$SEQ] PATH A step 4: upgrade 720p → 1080p"
    $PY $UPGRADE --seqs "$SEQ"

    # Step 5: validity copy (group → /scr)
    GROUP_VAL=$GROUP_ROOT/mask_validity/$SEQ
    SCR_VAL=$SCR_ROOT/mask_validity/$SEQ
    if [[ -d "$GROUP_VAL" ]]; then
        if [[ -d "$SCR_VAL" ]]; then
            # rename existing → .bak (only if no .bak yet)
            if [[ ! -d "${SCR_VAL}.bak" ]]; then
                echo "[$SEQ] PATH A step 5: bak existing /scr validity"
                mv "$SCR_VAL" "${SCR_VAL}.bak"
            else
                echo "[$SEQ] PATH A step 5: existing /scr validity .bak already present → leaving, removing current"
                rm -rf "$SCR_VAL"
            fi
        fi
        echo "[$SEQ] PATH A step 5: copy group validity → /scr"
        mkdir -p "$(dirname "$SCR_VAL")"
        cp -r "$GROUP_VAL" "$SCR_VAL"
    else
        echo "[$SEQ] PATH A step 5: no group validity for $SEQ → skipping validity copy"
    fi

else
    # ---- PATH B ----
    # Step 1: bak existing shard (cp -- inverse-perm rewrites in place atomically,
    #         but we want a forensic copy of the broken state)
    if [[ -d "$BAK_DIR" ]]; then
        echo "[$SEQ] .swap_fix.bak already exists — leaving prev intact"
    else
        echo "[$SEQ] PATH B step 1: cp -al $SHARD_DIR → $BAK_DIR (hardlink bak)"
        # Hardlink copy is fast and atomic per-file. Subsequent atomic os.replace
        # in restore script will detach the new file from the hardlink.
        cp -al "$SHARD_DIR" "$BAK_DIR"
    fi

    # Step 2: inverse-perm restore
    echo "[$SEQ] PATH B step 2: inverse-perm restore"
    $PY $RESTORE_B --seq "$SEQ"
fi

# Sentinel for restored state
touch "$SHARD_DIR/.restored_swap24"
echo "=== restore_one_seq $SEQ DONE rc=$? ==="
date
