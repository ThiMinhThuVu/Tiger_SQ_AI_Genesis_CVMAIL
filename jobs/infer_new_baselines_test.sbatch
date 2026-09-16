#!/usr/bin/env bash
#SBATCH --job-name=baseline-test-quant
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/baseline_test_quant_%j.log
#SBATCH --error=logs/baseline_test_quant_%j.log
set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate

python scripts/infer_foundation_test.py \
  --model dinov2 \
  --config configs/dinov2_vits14.yaml \
  --model-dir artifacts/new_baselines/dinov2_vits14 \
  --output artifacts/new_baselines/dinov2_vits14/test_quantitative.json

python scripts/infer_foundation_test.py \
  --model medsam \
  --config configs/medsam_vit_b.yaml \
  --model-dir artifacts/new_baselines/medsam_vit_b_multitask \
  --output artifacts/new_baselines/medsam_vit_b_multitask/test_quantitative.json

python scripts/run_nnunet_test.py
