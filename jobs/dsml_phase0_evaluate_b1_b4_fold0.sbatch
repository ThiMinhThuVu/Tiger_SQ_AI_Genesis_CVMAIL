#!/usr/bin/env bash
#SBATCH --job-name=dsml-p0-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-3%2
#SBATCH --output=logs/dsml_phase0_evaluate_b1_b4_fold0_%A_%a.log
#SBATCH --error=logs/dsml_phase0_evaluate_b1_b4_fold0_%A_%a.log

set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate

variants=(
  "mask2former_swin_small_b1_recall_reweight_fold0"
  "mask2former_swin_small_b2_sink_margin_uniform_fold0"
  "mask2former_swin_small_b3_sink_margin_symmetric_fold0"
  "mask2former_swin_small_b4_sink_margin_directed_fold0"
)
experiment_id="${variants[$SLURM_ARRAY_TASK_ID]}"
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
