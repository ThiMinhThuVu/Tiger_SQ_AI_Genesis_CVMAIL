#!/usr/bin/env bash
#SBATCH --job-name=tiger_d517_seg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a5000:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=full_version/center_stratified_32_8/logs/d517_seg_controller_%j.log
#SBATCH --error=full_version/center_stratified_32_8/logs/d517_seg_controller_%j.log
set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

manifest=full_version/center_stratified_32_8/splits/case_folds_d517.json
output_root=full_version/center_stratified_32_8/artifacts

# The cluster currently does not isolate CUDA devices reliably for separate
# array jobs. Use the two physical devices explicitly approved for this rerun.
selected_gpus=(1 3)
echo "Pinned physical GPUs: ${selected_gpus[*]}"

run_index() {
  local index="$1"
  local fold
  if (( index < 5 )); then
    fold="${index}"
    python full_version/task1_merged_16cls/scripts/train_coarse_only.py --mode train \
      --fold "${fold}" \
      --config full_version/center_stratified_32_8/configs/d517_task1_coarse_best.yaml \
      --split-manifest "${manifest}" --output-root "${output_root}"
  elif (( index < 10 )); then
    fold="$((index - 5))"
    python full_version/task2_fine_31cls/scripts/train_full.py \
      --fold "${fold}" \
      --config full_version/center_stratified_32_8/configs/d517_task2_fine_p0.yaml \
      --split-manifest "${manifest}" --output-root "${output_root}"
  else
    fold="$((index - 10))"
    python full_version/task2_fine_31cls/scripts/train_full.py \
      --fold "${fold}" \
      --config full_version/center_stratified_32_8/configs/d517_task2_fine_p2.yaml \
      --split-manifest "${manifest}" --output-root "${output_root}"
  fi
}

worker() {
  local gpu="$1"
  shift
  local index
  for index in "$@"; do
    local log="full_version/center_stratified_32_8/logs/d517_seg_${SLURM_JOB_ID}_${index}.log"
    echo "Starting index ${index} on physical GPU ${gpu}; log=${log}"
    CUDA_VISIBLE_DEVICES="${gpu}" run_index "${index}" >"${log}" 2>&1
    echo "Completed index ${index} on physical GPU ${gpu}"
  done
}

worker "${selected_gpus[0]}" 0 2 4 6 8 10 12 14 &
worker "${selected_gpus[1]}" 1 3 5 7 9 11 13 &
wait
