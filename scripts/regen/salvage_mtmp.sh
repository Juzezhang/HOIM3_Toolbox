#!/bin/bash
# Salvage completed-but-unswapped merge outputs: a crashed merge_pack_37.py (old NFS-rmtree
# bug) leaves a COMPLETE <seq>.mtmp (verified 37-view) but never swapped it in. This does the
# NFS-safe swap for every such .mtmp. Safe to re-run. The merge process must be DEAD first.
set -u
PACK=/simurgh2/datasets/HOI-M3/mhr_mono_packed
PY=/simurgh2/users/juze/anaconda3/envs/mvbodyfit/bin/python
shopt -s nullglob
for mt in "$PACK"/*.mtmp; do
  seq=$(basename "$mt" .mtmp); seqdir="$PACK/$seq"
  # staleness guard: skip a .mtmp touched in the last 180s — a live merge may still be
  # writing it (only crashed/finished jobs leave a quiescent .mtmp). Avoids racing.
  newest=$(find "$mt" -type f -newermt '-180 seconds' 2>/dev/null | head -1)
  [ -n "$newest" ] && { echo "[salvage] $seq: .mtmp written <180s ago -> skip (live job)"; continue; }
  # verify mtmp is a complete 37-view pack
  good=$($PY -c "
import numpy as np,glob,os,sys
mt='$mt'; ok=True; n=0
for p in sorted(glob.glob(mt+'/person*')):
    n+=1
    try:
        m=np.load(os.path.join(p,'meta.npz')); k=np.load(os.path.join(p,'keypoints2d_70.npy'),mmap_mode='r')
        for f in ('shape_params.npy','model_parameters.npy'):
            assert os.path.exists(os.path.join(p,f))
        if m['views'].shape[0]!=37 or k.shape[1]!=37: ok=False
    except Exception: ok=False
print('OK' if (ok and n>0) else 'BAD')")
  if [ "$good" != "OK" ]; then echo "[salvage] $seq: mtmp INCOMPLETE/BAD -> leave (rm? no)"; continue; fi
  # NFS-safe swap
  rm -rf "$seqdir.old" 2>/dev/null
  mv "$seqdir" "$seqdir.old" 2>/dev/null && mv "$mt" "$seqdir" && { rm -rf "$seqdir.old" 2>/dev/null; echo "merged37-salvaged" > "$seqdir/.pack_done"; echo "[salvage] $seq: SWAPPED IN (37-view) ✓"; } || echo "[salvage] $seq: swap FAILED"
done
