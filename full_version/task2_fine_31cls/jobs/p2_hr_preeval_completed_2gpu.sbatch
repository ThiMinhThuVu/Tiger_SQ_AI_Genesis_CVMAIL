#!/usr/bin/env bash
#SBATCH --job-name=tiger_p2_pretest
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a5000:2
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=72G
#SBATCH --time=12:00:00
#SBATCH --output=full_version/task2_fine_31cls/logs/p2_pretest_%j.log
#SBATCH --error=full_version/task2_fine_31cls/logs/p2_pretest_%j.log

set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

config="full_version/task2_fine_31cls/configs/p2_hr640x1120_seed2026.yaml"
split="full_version/task2_fine_31cls/splits/case_folds_v1.json"
model_dir="full_version/task2_fine_31cls/artifacts/p2_hr_pilot/p2_hr640x1120_seed2026"
oof_dir="full_version/task2_fine_31cls/artifacts/p2_hr_oof"

allocated_devices="${CUDA_VISIBLE_DEVICES}"
mapfile -t device_ids < <(
  nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits \
    | awk -F',' '{gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3); if ($2 >= 18000 && $3 <= 10) print $1, $2}' \
    | sort -k2,2nr \
    | head -2 \
    | awk '{print $1}'
)
if [[ "${#device_ids[@]}" -ne 2 ]]; then
  echo "Fewer than two idle physical GPUs have at least 18 GiB free" >&2
  nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits >&2
  exit 4
fi
echo "Slurm-visible GPUs=${allocated_devices}; selected idle physical GPUs=${device_ids[*]}"

eval_folds=(0 1 3 4)
for start in 0 2; do
  eval_pids=()
  batch_folds=()
  for offset in 0 1; do
    position=$((start + offset))
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
      > "full_version/task2_fine_31cls/logs/p2_pretest_${SLURM_JOB_ID}_fold${fold}.log" 2>&1 &
    eval_pids+=("$!")
    batch_folds+=("${fold}")
  done

  status=0
  for index in 0 1; do
    if ! wait "${eval_pids[$index]}"; then
      echo "Held-out evaluation fold ${batch_folds[$index]} failed" >&2
      status=1
    else
      echo "Held-out evaluation fold ${batch_folds[$index]} completed"
    fi
  done
  if [[ "${status}" -ne 0 ]]; then
    exit "${status}"
  fi
done

echo "Completed pre-evaluation for folds 0, 1, 3, and 4"
