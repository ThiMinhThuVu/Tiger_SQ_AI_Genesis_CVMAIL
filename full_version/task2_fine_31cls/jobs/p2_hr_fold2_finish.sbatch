#!/usr/bin/env bash
#SBATCH --job-name=tiger_p2_f2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=full_version/task2_fine_31cls/logs/p2_fold2_%j.log
#SBATCH --error=full_version/task2_fine_31cls/logs/p2_fold2_%j.log

set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

config="full_version/task2_fine_31cls/configs/p2_hr640x1120_seed2026.yaml"
split="full_version/task2_fine_31cls/splits/case_folds_v1.json"
training_root="full_version/task2_fine_31cls/artifacts/p2_hr_pilot"
model_dir="${training_root}/p2_hr640x1120_seed2026"
oof_dir="full_version/task2_fine_31cls/artifacts/p2_hr_oof"

python full_version/task2_fine_31cls/scripts/train_full.py \
  --config "${config}" --split-manifest "${split}" --fold 2 \
  --output-root "${training_root}"

python full_version/task2_fine_31cls/scripts/evaluate_test.py \
  --mode fold --fold 2 --config "${config}" --split-manifest "${split}" \
  --model-dir "${model_dir}" --output-dir "${oof_dir}"

python full_version/task2_fine_31cls/scripts/evaluate_test.py \
  --mode aggregate --config "${config}" --split-manifest "${split}" \
  --model-dir "${model_dir}" --output-dir "${oof_dir}"
