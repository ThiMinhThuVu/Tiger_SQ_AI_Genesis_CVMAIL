#!/usr/bin/env bash
#SBATCH --job-name=tiger_p2_full2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a5000:2
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=12
#SBATCH --mem=110G
#SBATCH --time=48:00:00
#SBATCH --output=full_version/task2_fine_31cls/logs/p2_full_%j.log
#SBATCH --error=full_version/task2_fine_31cls/logs/p2_full_%j.log

set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

config="full_version/task2_fine_31cls/configs/p2_hr640x1120_seed2026.yaml"
split="full_version/task2_fine_31cls/splits/case_folds_v1.json"
training_root="full_version/task2_fine_31cls/artifacts/p2_hr_pilot"
model_dir="${training_root}/p2_hr640x1120_seed2026"
oof_dir="full_version/task2_fine_31cls/artifacts/p2_hr_oof"

allocated_devices="${CUDA_VISIBLE_DEVICES}"
mapfile -t device_ids < <(
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F',' '{gsub(/ /, "", $1); gsub(/ /, "", $2); if ($2 >= 22000) print $1, $2}' \
    | sort -k2,2nr \
    | head -2 \
    | awk '{print $1}'
)
if [[ "${#device_ids[@]}" -ne 2 ]]; then
  echo "Fewer than two physical GPUs have at least 22 GiB free" >&2
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits >&2
  exit 4
fi
echo "Slurm-visible GPUs=${allocated_devices}; selected physical GPUs=${device_ids[*]}"

train_folds=(2 3)
train_pids=()
for index in 0 1; do
  fold="${train_folds[$index]}"
  device_id="${device_ids[$index]}"
  echo "Training fold=${fold} on physical GPU=${device_id}"
  CUDA_VISIBLE_DEVICES="${device_id}" python \
    full_version/task2_fine_31cls/scripts/train_full.py \
    --config "${config}" \
    --split-manifest "${split}" \
    --fold "${fold}" \
    --output-root "${training_root}" \
    > "full_version/task2_fine_31cls/logs/p2_full_${SLURM_JOB_ID}_train_fold${fold}.log" 2>&1 &
  train_pids+=("$!")
done

train_status=0
for index in 0 1; do
  if ! wait "${train_pids[$index]}"; then
    echo "Training fold ${train_folds[$index]} failed" >&2
    train_status=1
  else
    echo "Training fold ${train_folds[$index]} completed"
  fi
done
if [[ "${train_status}" -ne 0 ]]; then
  exit "${train_status}"
fi

eval_folds=(0 1 2 3 4)
for start in 0 2 4; do
  eval_pids=()
  eval_batch_folds=()
  for offset in 0 1; do
    position=$((start + offset))
    if [[ "${position}" -ge "${#eval_folds[@]}" ]]; then
      continue
    fi
    fold="${eval_folds[$position]}"
    device_id="${device_ids[$offset]}"
    echo "Evaluating held-out fold=${fold} on physical GPU=${device_id}"
    CUDA_VISIBLE_DEVICES="${device_id}" python \
      full_version/task2_fine_31cls/scripts/evaluate_test.py \
      --mode fold \
      --fold "${fold}" \
      --config "${config}" \
      --split-manifest "${split}" \
      --model-dir "${model_dir}" \
      --output-dir "${oof_dir}" \
      > "full_version/task2_fine_31cls/logs/p2_full_${SLURM_JOB_ID}_eval_fold${fold}.log" 2>&1 &
    eval_pids+=("$!")
    eval_batch_folds+=("${fold}")
  done

  eval_status=0
  for index in "${!eval_pids[@]}"; do
    if ! wait "${eval_pids[$index]}"; then
      echo "Held-out evaluation fold ${eval_batch_folds[$index]} failed" >&2
      eval_status=1
    else
      echo "Held-out evaluation fold ${eval_batch_folds[$index]} completed"
    fi
  done
  if [[ "${eval_status}" -ne 0 ]]; then
    exit "${eval_status}"
  fi
done

echo "Aggregating full five-fold OOF"
python full_version/task2_fine_31cls/scripts/evaluate_test.py \
  --mode aggregate \
  --config "${config}" \
  --split-manifest "${split}" \
  --model-dir "${model_dir}" \
  --output-dir "${oof_dir}"
