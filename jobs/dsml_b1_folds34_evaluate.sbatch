#!/usr/bin/env bash
#SBATCH --job-name=dsml-b1-f34-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-1%2
#SBATCH --output=logs/dsml_b1_folds34_evaluate_%A_%a.log
#SBATCH --error=logs/dsml_b1_folds34_evaluate_%A_%a.log

set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate

configs=(
  "mask2former_swin_small_b1_recall_reweight_fold3"
  "mask2former_swin_small_b1_recall_reweight_fold4"
)
target_folds=(3 4)
experiment_id="${configs[$SLURM_ARRAY_TASK_ID]}"
target_fold="${target_folds[$SLURM_ARRAY_TASK_ID]}"
model_dir="artifacts/dsml_phase0_b1_folds34/${experiment_id}"
output="${model_dir}/test_task1_quantitative_fold${target_fold}.json"

if [[ -e "${output}" ]]; then
  echo "Refusing to overwrite existing ${output}" >&2
  exit 2
fi

python scripts/infer_mask2former_improved_test.py \
  --config "configs/${experiment_id}.yaml" \
  --model-dir "${model_dir}" \
  --output "${output}" \
  --folds "${target_fold}" \
  --allow-incomplete
