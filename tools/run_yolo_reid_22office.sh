#!/bin/bash
# Generate masks for 22 office sequences + livingroom_data21 using YOLO+ReID
# Then run SAM-3D-Body mono + VitPose
set -e

cd /simurgh/u/juze/code/HOIM3_Toolbox
PYTHON="/simurgh2/users/juze/anaconda3/envs/HOIM3_Toolbox/bin/python"

ALL_SEQS=(
  livingroom_data21
  office_data24 office_data25 office_data26 office_data27 office_data28 office_data29
  office_data31 office_data32 office_data33 office_data34 office_data35 office_data36
  office_data37 office_data38 office_data39 office_data40 office_data41 office_data42
  office_data43 office_data44 office_data50 office_data55
)

N=${#ALL_SEQS[@]}
CHUNK=$(( (N + 3) / 4 ))

echo "$(date) Starting YOLO+ReID for $N sequences on 4 GPUs"

# Split sequences across 4 GPUs
for gpu in 0 1 2 3; do
  start=$((gpu * CHUNK))
  end=$((start + CHUNK))
  [ $end -gt $N ] && end=$N
  [ $start -ge $N ] && continue

  gpu_seqs=("${ALL_SEQS[@]:$start:$((end - start))}")
  echo "$(date) GPU $gpu: ${gpu_seqs[*]}"

  $PYTHON scripts/yolo_seg/inference_masks.py \
    --gpu $gpu \
    --sequences ${gpu_seqs[*]} \
    > /tmp/yolo_reid_gpu${gpu}.log 2>&1 &
done

wait
echo "$(date) YOLO+ReID ALL DONE"

# Stage 2: Run SAM-3D-Body mono
echo "$(date) Starting SAM-3D-Body mono..."
cd /simurgh/u/juze/code/fast-sam-3d-body
PYTHON_SAM="/simurgh2/users/juze/anaconda3/envs/fastsam3d/bin/python"

$PYTHON_SAM tools/process_hoim3_npz_masks.py \
  --checkpoint_path checkpoints/sam-3d-body-dinov3/model.ckpt \
  --mhr_path checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt \
  --batch_size 128 \
  --inference_type body \
  --gpus 0,1,2,3 \
  --sequences ${ALL_SEQS[*]} 2>&1 | tee /tmp/mono_22office.log

echo "$(date) Mono ALL DONE"

# Stage 3: VitPose
echo "$(date) Starting VitPose..."
cd /simurgh/u/juze/code/mv-bodyfit
PYTHON_VP="/simurgh2/users/juze/anaconda3/envs/mvbodyfit/bin/python"
MONO="/scr/juze/datasets/HOI-M3/mhr_mono"
LOCKDIR="/tmp/vitpose_locks_22office"
mkdir -p "$LOCKDIR"

run_vitpose_gpu() {
  local gpu=$1
  for seq in "${ALL_SEQS[@]}"; do
    # Skip if done
    [ -d "/scr/juze/datasets/HOI-M3/vitpose/$seq" ] && \
      [ "$(find /scr/juze/datasets/HOI-M3/vitpose/$seq -name 'keypoints_coco23*.npy' 2>/dev/null | head -1)" ] && continue
    # Skip if no mono
    [ "$(find "$MONO/$seq" -name '*.npz' -maxdepth 3 2>/dev/null | head -1)" ] || continue
    # Lock
    mkdir "$LOCKDIR/${seq}.lock" 2>/dev/null || continue
    echo "$(date) GPU $gpu: VitPose $seq"
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON_VP tools/precompute_vitpose_hoim3.py \
      --sequence "$seq" --mono_root "$MONO" 2>&1
    echo "$(date) GPU $gpu: VitPose DONE $seq"
  done
}

for g in 0 1 2 3; do
  run_vitpose_gpu $g &
done
wait
rm -rf "$LOCKDIR"
echo "$(date) VitPose ALL DONE"
echo "$(date) Full pipeline complete for ${#ALL_SEQS[@]} sequences"
