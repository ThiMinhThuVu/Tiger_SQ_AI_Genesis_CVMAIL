#!/usr/bin/env bash
#SBATCH --job-name=tiger_cs_t1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-4%2
#SBATCH --output=full_version/center_stratified_32_8/logs/task1_%A_%a.log
#SBATCH --error=full_version/center_stratified_32_8/logs/task1_%A_%a.log
set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate
python full_version/task1_merged_16cls/scripts/train_coarse_only.py --mode train \
  --fold "${SLURM_ARRAY_TASK_ID}" \
  --config full_version/center_stratified_32_8/configs/task1_coarse_best.yaml \
  --split-manifest full_version/center_stratified_32_8/splits/case_folds.json \
  --output-root full_version/center_stratified_32_8/artifacts

