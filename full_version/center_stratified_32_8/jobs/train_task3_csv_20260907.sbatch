#!/usr/bin/env bash
#SBATCH --job-name=tiger_cs_t3_csv2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=full_version/center_stratified_32_8/logs/task3_csv_20260907_%j.log
#SBATCH --error=full_version/center_stratified_32_8/logs/task3_csv_20260907_%j.log
set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate
python full_version/center_stratified_32_8/scripts/train_task3.py \
  --config full_version/center_stratified_32_8/configs/task2_fine_best.yaml \
  --manifest full_version/center_stratified_32_8/splits/case_folds.json \
  --model-dir full_version/center_stratified_32_8/artifacts/center_strat32x8_task2_fine_improved_v2 \
  --output-root full_version/center_stratified_32_8/artifacts/task3_visibility_csv_20260907
