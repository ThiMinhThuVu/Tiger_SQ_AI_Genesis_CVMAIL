#!/usr/bin/env bash
#SBATCH --job-name=c5-rbcot-f0-test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/mask2former_c5_rb_cot_fold0_test_%j.log
#SBATCH --error=logs/mask2former_c5_rb_cot_fold0_test_%j.log

set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate

model_dir="artifacts/mask2former_research/mask2former_swin_small_c5_rb_cot_bs4_task1"
output="${model_dir}/test_fold0_quantitative.json"
if [[ -e "${output}" ]]; then
  echo "Refusing to overwrite existing ${output}" >&2
  exit 2
fi

python scripts/infer_mask2former_improved_test.py \
  --config configs/mask2former_swin_small_c5_rb_cot_bs4_task1.yaml \
  --model-dir "${model_dir}" \
  --folds 0 \
  --output "${output}"
