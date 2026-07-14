#!/bin/bash
# Per-seq concat+commit: when a seq's DenseFit window array has produced all its
# windows, stitch -> savgol -> convert(16 betas) -> BACKUP MHR -> merge into
# smplx_with_distortion/. Serialized (one concat at a time, ~3min each).
set -uo pipefail
ENV=/simurgh2/users/juze/anaconda3/envs/densefit/bin/python
MM=${DENSEFIT_ROOT:-/path/to/densefit_workspace}
SMPLXWD=/simurgh2/datasets/HOI-M3/smplx_with_distortion
LOG=/simurgh2/users/juze/calibjoint/concat_orch.log
W=300
SEQS=(livingroom_data03 livingroom_data04 livingroom_data30 livingroom_data31 livingroom_data32 livingroom_data48 livingroom_data07 livingroom_data11 livingroom_data21 livingroom_data33 livingroom_data34 bedroom_data05 bedroom_data35 bedroom_data01 bedroom_data03 bedroom_data15 bedroom_data30 bedroom_data31 bedroom_data32 bedroom_data33 bedroom_data34 diningroom_data09 diningroom_data11 diningroom_data01 diningroom_data06 office_data18 office_data61 office_data03 office_data25 office_data26 office_data27 office_data29 office_data30 office_data31 office_data37 office_data38 office_data39 office_data43 office_data44 office_data45 office_data50)
iter=0
while [ $iter -lt 3000 ]; do
  iter=$((iter+1)); pending=0
  for s in "${SEQS[@]}"; do
    # already committed as DenseFit?
    if [ -f "$SMPLXWD/${s}_person0_meta.json" ] && grep -q '"DenseFit' "$SMPLXWD/${s}_person0_meta.json" 2>/dev/null; then continue; fi
    [ -f /simurgh2/users/juze/calibjoint/prepinfo_$s.json ] || { pending=1; continue; }
    NF=$($ENV -c "import json;print(json.load(open('/simurgh2/users/juze/calibjoint/prepinfo_$s.json'))['frames'])" 2>/dev/null) || { pending=1; continue; }
    NWIN=$(( (NF + W - 1) / W ))
    have=$(find $MM/output/ma_3d/${s}_full_w* -name 'smplx_params_body_id-00.npz' 2>/dev/null | wc -l)
    if [ "$have" -ge "$NWIN" ]; then
      echo "$(date +%H:%M) [concat] $s: $have/$NWIN windows -> commit" | tee -a "$LOG"
      $ENV /simurgh2/users/juze/calibjoint/densefit_concat.py "$s" $W --commit >>"$LOG" 2>&1 \
        && echo "$(date +%H:%M) [concat] $s COMMITTED" | tee -a "$LOG" \
        || echo "$(date +%H:%M) [concat] $s FAILED" | tee -a "$LOG"
    else
      pending=1
    fi
  done
  [ $pending -eq 0 ] && { echo "$(date +%H:%M) ALL 41 CONCAT+COMMITTED" | tee -a "$LOG"; break; }
  sleep 180
done
