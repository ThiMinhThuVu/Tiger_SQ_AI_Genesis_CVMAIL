#!/usr/bin/env bash
#SBATCH --job-name=dsml-a0-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/dsml_phase0_a0_matched_env_evaluate_fold0_%j.log
#SBATCH --error=logs/dsml_phase0_a0_matched_env_evaluate_fold0_%j.log

set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate

model_dir="artifacts/dsml_phase0/mask2former_swin_small_c4_a0_nocontext_bs4_task1"
output="${model_dir}/test_task1_quantitative_fold0.json"
if [[ -e "${output}" ]]; then
  echo "Refusing to overwrite existing ${output}" >&2
  exit 2
fi

python scripts/infer_mask2former_improved_test.py \
  --config configs/mask2former_swin_small_c4_a0_nocontext_bs4_task1.yaml \
  --model-dir "${model_dir}" \
  --output "${output}" \
  --folds 0 \
  --allow-incomplete
