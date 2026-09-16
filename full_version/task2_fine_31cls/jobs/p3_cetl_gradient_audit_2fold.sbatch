#!/usr/bin/env bash
#SBATCH --job-name=tiger_p3_grad2
#SBATCH --partition=gpu
# Request one spare because this node has long-lived processes outside Slurm.
# Only two clean allocated GPUs are used.
#SBATCH --gres=gpu:a5000:3
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=full_version/task2_fine_31cls/logs/p3_grad_%j.log
#SBATCH --error=full_version/task2_fine_31cls/logs/p3_grad_%j.log

set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

IFS=',' read -r -a allocated_device_ids <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#allocated_device_ids[@]}" -ne 3 ]]; then
  echo "Expected exactly 3 allocated GPUs, got: ${CUDA_VISIBLE_DEVICES}" >&2
  exit 3
fi
device_candidates=()
for device_id in "${allocated_device_ids[@]}"; do
  free_memory="$(nvidia-smi -i "${device_id}" --query-gpu=memory.free --format=csv,noheader,nounits | head -1)"
  device_candidates+=("${free_memory}:${device_id}")
done
mapfile -t device_candidates < <(
  printf '%s\n' "${device_candidates[@]}" | sort -t: -k1,1nr
)
device_ids=()
for candidate in "${device_candidates[@]}"; do
  free_memory="${candidate%%:*}"
  device_id="${candidate##*:}"
  if [[ "${free_memory}" -ge 22000 ]]; then
    device_ids+=("${device_id}")
  fi
  if [[ "${#device_ids[@]}" -eq 2 ]]; then
    break
  fi
done
if [[ "${#device_ids[@]}" -ne 2 ]]; then
  echo "Fewer than two allocated GPUs have at least 22 GiB free" >&2
  printf 'candidate=%s\n' "${device_candidates[@]}" >&2
  exit 4
fi

base_config="full_version/task2_fine_31cls/configs/p2_hr640x1120_seed2026.yaml"
cetl_config="full_version/task2_fine_31cls/configs/p3_hr640x1120_cetl_w025_seed2026.yaml"
split="full_version/task2_fine_31cls/splits/case_folds_v1.json"
checkpoint_root="full_version/task2_fine_31cls/artifacts/p2_hr_pilot/p2_hr640x1120_seed2026"
output_root="full_version/task2_fine_31cls/artifacts/p3_cetl_gradient_audit"
folds=(0 1)
pids=()

for index in 0 1; do
  device_id="${device_ids[$index]}"
  fold="${folds[$index]}"
  echo "Launching gradient audit fold=${fold} on physical GPU=${device_id}"
  CUDA_VISIBLE_DEVICES="${device_id}" python \
    full_version/task2_fine_31cls/scripts/p3_cetl_gradient_audit.py \
    --base-config "${base_config}" \
    --cetl-config "${cetl_config}" \
    --split-manifest "${split}" \
    --checkpoint-root "${checkpoint_root}" \
    --fold "${fold}" \
    --batches 8 \
    --batch-size 2 \
    --output "${output_root}/fold_${fold}.json" \
    > "full_version/task2_fine_31cls/logs/p3_grad_${SLURM_JOB_ID}_fold${fold}.log" 2>&1 &
  pids+=("$!")
done

status=0
for index in 0 1; do
  if ! wait "${pids[$index]}"; then
    echo "Gradient audit fold ${folds[$index]} failed" >&2
    status=1
  else
    echo "Gradient audit fold ${folds[$index]} completed"
  fi
done
exit "${status}"
