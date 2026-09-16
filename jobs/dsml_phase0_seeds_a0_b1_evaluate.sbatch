#!/usr/bin/env bash
#SBATCH --job-name=dsml-seeds-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-3%2
#SBATCH --output=logs/dsml_phase0_seeds_a0_b1_evaluate_%A_%a.log
#SBATCH --error=logs/dsml_phase0_seeds_a0_b1_evaluate_%A_%a.log

set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate

configs=(
  "mask2former_swin_small_c4_a0_nocontext_bs4_task1_seed2027"
  "mask2former_swin_small_c4_a0_nocontext_bs4_task1_seed2028"
  "mask2former_swin_small_b1_recall_reweight_fold0_seed2027"
  "mask2former_swin_small_b1_recall_reweight_fold0_seed2028"
)
experiment_id="${configs[$SLURM_ARRAY_TASK_ID]}"
config="configs/${experiment_id}.yaml"
model_dir="artifacts/dsml_phase0/${experiment_id}"
output="${model_dir}/test_task1_quantitative_fold0.json"

if [[ -e "${output}" ]]; then
  echo "Refusing to overwrite existing ${output}" >&2
  exit 2
fi

echo "Evaluating ${experiment_id} (array task ${SLURM_ARRAY_TASK_ID}) on held-out test fold 0"
python scripts/infer_mask2former_improved_test.py \
  --config "${config}" \
  --model-dir "${model_dir}" \
  --output "${output}" \
  --folds 0 \
  --allow-incomplete
