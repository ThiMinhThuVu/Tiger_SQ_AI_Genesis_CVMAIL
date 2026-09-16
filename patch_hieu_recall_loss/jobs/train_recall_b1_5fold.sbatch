#!/bin/bash
#SBATCH --job-name=tiger_recall_b1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a5000:2
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-4%2
#SBATCH --output=full_version/task2_fine_31cls/logs/train_recall_b1_%A_%a.log
#SBATCH --error=full_version/task2_fine_31cls/logs/train_recall_b1_%A_%a.log

# Recall Loss (B1) clean rerun on the pinned repo .venv (torch 2.7.1+cu128).
# Per-fold config carries the fold's OOF-only miss-rate vector. Same OT/lr/
# schedule/split as C4-A0 -- single-factor change.
#
# PILOT-FIRST: watch fold 0's held-out test result vs C4-A0 fold-0 OOF
# (fine task 0.783868) before trusting folds 1-4. B1 Phase-0 hit 0.7944 but
# under the wrong torch; this rerun is what makes it interpretable.

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
fold="${SLURM_ARRAY_TASK_ID}"
echo "Recall-B1 fold=${fold}; physical GPU ${selected_device} with ${selected_free} MiB free"

python full_version/task2_fine_31cls/scripts/train_full.py \
  --config "full_version/task2_fine_31cls/configs/full40_fine31_recall_b1_fold${fold}_seed2026.yaml" \
  --split-manifest full_version/task2_fine_31cls/splits/case_folds_v1.json \
  --fold "${fold}" \
  --output-root full_version/task2_fine_31cls/artifacts/seed_2026
