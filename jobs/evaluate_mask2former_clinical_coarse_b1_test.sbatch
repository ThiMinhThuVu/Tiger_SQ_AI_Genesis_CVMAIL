#!/usr/bin/env bash
#SBATCH --job-name=m2f-coarse-test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/mask2former_clinical_coarse_b1_test_%j.log
#SBATCH --error=logs/mask2former_clinical_coarse_b1_test_%j.log
set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate
python scripts/infer_mask2former_improved_test.py \
  --config configs/mask2former_swin_small_clinical_coarse_b1.yaml \
  --model-dir artifacts/mask2former_improved/mask2former_swin_small_clinical_coarse_b1 \
  --output artifacts/mask2former_improved/mask2former_swin_small_clinical_coarse_b1/test_quantitative.json
