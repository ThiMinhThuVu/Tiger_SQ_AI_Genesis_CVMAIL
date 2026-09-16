#!/usr/bin/env bash
#SBATCH --job-name=c3-ot-f0-test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/mask2former_c3_ot_fold0_test_%j.log
#SBATCH --error=logs/mask2former_c3_ot_fold0_test_%j.log

set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

output="artifacts/mask2former_research/mask2former_swin_small_c3_ot_task1/test_fold0_quantitative.json"
if [[ -e "${output}" ]]; then
  echo "Refusing to overwrite existing ${output}" >&2
  exit 2
fi

python scripts/infer_mask2former_improved_test.py \
  --config configs/mask2former_swin_small_c3_ot_task1.yaml \
  --model-dir artifacts/mask2former_research/mask2former_swin_small_c3_ot_task1 \
  --folds 0 \
  --output "${output}"
