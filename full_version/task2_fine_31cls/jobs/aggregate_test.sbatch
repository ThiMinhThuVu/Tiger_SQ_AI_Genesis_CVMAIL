#!/usr/bin/env bash
#SBATCH --job-name=tiger_f31_agg
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=full_version/task2_fine_31cls/logs/test_aggregate_%j.log
#SBATCH --error=full_version/task2_fine_31cls/logs/test_aggregate_%j.log

set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate

python full_version/task2_fine_31cls/scripts/evaluate_test.py \
  --mode aggregate \
  --config full_version/task2_fine_31cls/configs/full40_fine31_c4_a0_seed2026.yaml \
  --split-manifest full_version/task2_fine_31cls/splits/case_folds_v1.json \
  --model-dir full_version/task2_fine_31cls/artifacts/seed_2026/full40_fine31_c4_a0_seed2026 \
  --output-dir full_version/task2_fine_31cls/artifacts/seed_2026/test_oof
