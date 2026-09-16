#!/usr/bin/env bash
#SBATCH --job-name=tiger_p2_hr3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a5000:3
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=18
#SBATCH --mem=150G
#SBATCH --time=48:00:00
#SBATCH --output=full_version/task2_fine_31cls/logs/p2_hr_%j.log
#SBATCH --error=full_version/task2_fine_31cls/logs/p2_hr_%j.log

set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

IFS=',' read -r -a device_ids <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#device_ids[@]}" -ne 3 ]]; then
  echo "Expected exactly 3 allocated GPUs, got: ${CUDA_VISIBLE_DEVICES}" >&2
  exit 3
fi

folds=(0 1 4)
pids=()
for index in 0 1 2; do
  device_id="${device_ids[$index]}"
  fold="${folds[$index]}"
  free_memory="$(nvidia-smi -i "${device_id}" --query-gpu=memory.free --format=csv,noheader,nounits | head -1)"
  if [[ "${free_memory}" -lt 22000 ]]; then
    echo "GPU ${device_id} has only ${free_memory} MiB free" >&2
    exit 4
  fi
  echo "Launching fold=${fold} on physical GPU=${device_id}, free=${free_memory} MiB"
  CUDA_VISIBLE_DEVICES="${device_id}" python \
    full_version/task2_fine_31cls/scripts/train_full.py \
    --config full_version/task2_fine_31cls/configs/p2_hr640x1120_seed2026.yaml \
    --split-manifest full_version/task2_fine_31cls/splits/case_folds_v1.json \
    --fold "${fold}" \
    --output-root full_version/task2_fine_31cls/artifacts/p2_hr_pilot \
    > "full_version/task2_fine_31cls/logs/p2_hr_${SLURM_JOB_ID}_fold${fold}.log" 2>&1 &
  pids+=("$!")
done

status=0
for index in 0 1 2; do
  if ! wait "${pids[$index]}"; then
    echo "Fold ${folds[$index]} failed; inspect its dedicated log" >&2
    status=1
  else
    echo "Fold ${folds[$index]} completed"
  fi
done
exit "${status}"
