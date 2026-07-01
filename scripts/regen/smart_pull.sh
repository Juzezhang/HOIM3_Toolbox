#!/bin/bash
# Smart pull: instead of rsync -r over the entire mhr_mono tree (44 min listing per cycle),
# list GCS sentinels (~thousand files, ~30s), then targeted gsutil cp -r for each
# (seq, view) whose sentinel exists on GCS but not locally.
#
# Runs in a loop, sleeping between cycles. Designed for fast catchup (~min per cycle).

set -u
GCS_BASE=gs://data-storage-0/HOI-M3/mhr_mono
NFS=/simurgh2/datasets/HOI-M3/mhr_mono
GSUTIL=/simurgh2/users/juze/anaconda3/envs/SAMPA/bin/gsutil
export BOTO_CONFIG=/simurgh/u/juze/.boto_gcs

LOG=/scr/juze/swap24_cleanup_logs/smart_pull.log
SLEEP=120
PARALLEL=20

mkdir -p "$NFS"
exec >> "$LOG" 2>&1
echo "$(date) [smart_pull] START pid=$$ parallel=$PARALLEL sleep=${SLEEP}s"

while true; do
  cycle_start=$(date +%s)
  echo "$(date) [smart_pull] cycle start — listing GCS sentinels"

  # Get sentinel paths from GCS (one per (seq, view))
  $GSUTIL ls "$GCS_BASE/*/*/.view_done_*" 2>/dev/null \
    | awk -F'/' '{print $(NF-2), $(NF-1)}' \
    | sort -u > /tmp/smart_pull_gcs.txt

  # Get sentinels we have locally
  /usr/bin/find "$NFS" -maxdepth 3 -name '.view_done_*' 2>/dev/null \
    | awk -F'/' '{print $(NF-2), $(NF-1)}' \
    | sort -u > /tmp/smart_pull_local.txt

  # Diff: on GCS but not local → must pull
  comm -23 /tmp/smart_pull_gcs.txt /tmp/smart_pull_local.txt > /tmp/smart_pull_todo.txt
  todo=$(wc -l < /tmp/smart_pull_todo.txt)
  echo "$(date) [smart_pull] GCS sentinels: $(wc -l < /tmp/smart_pull_gcs.txt), local: $(wc -l < /tmp/smart_pull_local.txt), to pull: $todo"

  if [ "$todo" -gt 0 ]; then
    # For each (seq, view) needing pull: gsutil -m cp -r entire view dir
    # Each line in todo is "seq view". For each:
    # 1. Check if hai-style tar exists at <seq>/<view>.tar — if yes, cp + extract
    # 2. Else fall back to cp -r of <seq>/<view>/ dir-of-npz (sc-style)
    cat /tmp/smart_pull_todo.txt | xargs -P "$PARALLEL" -I {} bash -c '
      line="$1"
      seq="${line% *}"
      view="${line##* }"
      tar_src="'$GCS_BASE'/$seq/$view.tar"
      dir_src="'$GCS_BASE'/$seq/$view"
      dst_view="'$NFS'/$seq/$view"
      mkdir -p "$dst_view"
      # Try tar first (hai-style, faster)
      if '$GSUTIL' -q stat "$tar_src" >/dev/null 2>&1; then
        # Stream tar from GCS, extract locally — avoids intermediate file.
        tmp_tar=$(mktemp /tmp/pull_${seq}_${view}_XXXX.tar)
        '$GSUTIL' -q cp "$tar_src" "$tmp_tar" && \
          tar -xf "$tmp_tar" -C "'$NFS'/$seq/" && \
          rm -f "$tmp_tar" && \
          echo "[pull-tar-ok $seq v$view]" || \
          { echo "[pull-tar-err $seq v$view]"; rm -f "$tmp_tar"; }
      else
        # Fall back to dir cp (sc-style)
        out=$('$GSUTIL' -m -q cp -r "$dir_src" "'$NFS'/$seq/" 2>&1)
        rc=$?
        if [ $rc -ne 0 ] && [ -n "$out" ]; then
          echo "[pull-dir-err $seq v$view rc=$rc] $(echo $out | head -c 120)"
        else
          echo "[pull-dir-ok $seq v$view]"
        fi
      fi
    ' _ {}
  fi

  dt=$(($(date +%s) - cycle_start))
  echo "$(date) [smart_pull] cycle done in ${dt}s, sleeping ${SLEEP}s"
  sleep "$SLEEP"
done
