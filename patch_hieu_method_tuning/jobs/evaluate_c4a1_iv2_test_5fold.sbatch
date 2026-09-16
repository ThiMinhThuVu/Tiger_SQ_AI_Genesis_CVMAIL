#!/usr/bin/env bash
#SBATCH --job-name=tiger_iv2_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a5000:4
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-4%2
#SBATCH --output=full_version/task2_fine_31cls/logs/test_c4a1_iv2_%A_%a.log
#SBATCH --error=full_version/task2_fine_31cls/logs/test_c4a1_iv2_%A_%a.log

set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

allocated_devices="${CUDA_VISIBLE_DEVICES}"
IFS="," read -r -a device_ids <<< "${allocated_devices}"
selected_device=""; selected_free=-1
for device_id in "${device_ids[@]}"; do
  free_memory="$(nvidia-smi -i "${device_id}" --query-gpu=memory.free --format=csv,noheader,nounits | head -1)"
  if (( free_memory > selected_free )); then selected_device="${device_id}"; selected_free="${free_memory}"; fi
done
if [[ -z "${selected_device}" || "${selected_free}" -lt 18000 ]]; then
  echo "No allocated GPU has at least 18 GiB free: ${allocated_devices}" >&2; exit 3
fi
export CUDA_VISIBLE_DEVICES="${selected_device}"
echo "Test fold=${SLURM_ARRAY_TASK_ID}; physical GPU ${selected_device} with ${selected_free} MiB free"

python full_version/task2_fine_31cls/scripts/evaluate_test.py \
  --mode fold \
  --fold "${SLURM_ARRAY_TASK_ID}" \
  --config full_version/task2_fine_31cls/configs/full40_fine31_c4a1_improved_v2_seed2026.yaml \
  --split-manifest full_version/task2_fine_31cls/splits/case_folds_v1.json \
  --model-dir full_version/task2_fine_31cls/artifacts/seed_2026/full40_fine31_c4a1_improved_v2_seed2026 \
  --output-dir full_version/task2_fine_31cls/artifacts/seed_2026/test_oof_c4a1_iv2
