#!/usr/bin/env bash
#SBATCH --job-name=tiger_d517_e1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=full_version/center_stratified_32_8/logs/d517_eval_task1_%j.log
#SBATCH --error=full_version/center_stratified_32_8/logs/d517_eval_task1_%j.log
set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate
python full_version/center_stratified_32_8/scripts/evaluate_validation.py --task task1 \
  --config full_version/center_stratified_32_8/configs/d517_task1_coarse_best.yaml \
  --manifest full_version/center_stratified_32_8/splits/case_folds_d517.json \
  --model-dir full_version/center_stratified_32_8/artifacts/center_strat32x8_d517_task1_coarse_best \
  --output full_version/center_stratified_32_8/reports/d517_task1_validation.json
