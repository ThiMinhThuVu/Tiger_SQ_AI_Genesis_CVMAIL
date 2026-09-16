#!/bin/bash
#SBATCH --job-name=tiger_iv2_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=full_version/task2_fine_31cls/logs/smoke_c4a1_iv2_%j.log
#SBATCH --error=full_version/task2_fine_31cls/logs/smoke_c4a1_iv2_%j.log

# Fold-0 3-epoch smoke for the Improved-v2 port (coarse+hierarchy+presence heads
# on the C4-A0 OT baseline). Verifies loading / fwd+bwd / val metric / checkpoint
# / output isolation before the 5-fold array. Inspect the log, then submit
# train_task2_c4a1_improved_v2.sbatch.

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
if [[ -z "${selected_device}" || "${selected_free}" -lt 22000 ]]; then
  echo "No allocated GPU has at least 22 GiB free: ${allocated_devices}" >&2; exit 3
fi
export CUDA_VISIBLE_DEVICES="${selected_device}"
echo "Smoke fold 0; physical GPU ${selected_device} with ${selected_free} MiB free"

python full_version/task2_fine_31cls/scripts/train_full.py \
  --config full_version/task2_fine_31cls/configs/full40_fine31_c4a1_improved_v2_seed2026.yaml \
  --split-manifest full_version/task2_fine_31cls/splits/case_folds_v1.json \
  --fold 0 --smoke \
  --output-root full_version/task2_fine_31cls/artifacts/smoke_c4a1_iv2
