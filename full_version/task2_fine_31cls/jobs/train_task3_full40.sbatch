#!/usr/bin/env bash
#SBATCH --job-name=tiger_t3_full40
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=full_version/task2_fine_31cls/logs/task3_full40_%j.log
#SBATCH --error=full_version/task2_fine_31cls/logs/task3_full40_%j.log

set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate

python scripts/train_mask2former_visibility.py \
  --config full_version/task2_fine_31cls/configs/full40_fine31_c4_a0_seed2026.yaml \
  --split-manifest full_version/task2_fine_31cls/splits/case_folds_v1.json \
  --model-dir full_version/task2_fine_31cls/artifacts/seed_2026/full40_fine31_c4_a0_seed2026 \
  --segmentation-results full_version/task2_fine_31cls/artifacts/seed_2026/test_oof/oof_test_metrics.json \
  --output-root full_version/task3_visibility/artifacts/full40_seed2026
