#!/usr/bin/env bash
#SBATCH --job-name=m2f-swin-small-full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/mask2former_swin_small_full_5fold_%j.log
#SBATCH --error=logs/mask2former_swin_small_full_5fold_%j.log
set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate

python scripts/train_mask2former.py \
  --config configs/mask2former_swin_small.yaml \
  --run-all-folds

python scripts/infer_mask2former_test.py \
  --config configs/mask2former_swin_small.yaml \
  --model-dir artifacts/mask2former/mask2former_swin_small \
  --output artifacts/mask2former/mask2former_swin_small/test_quantitative.json

python scripts/compare_mask2former_backbones.py \
  --candidate artifacts/mask2former/mask2former_swin_small/test_quantitative.json \
  --baseline artifacts/mask2former/mask2former_swin_tiny/test_quantitative.json \
  --output artifacts/mask2former/mask2former_swin_small/comparison_vs_swin_tiny.json
