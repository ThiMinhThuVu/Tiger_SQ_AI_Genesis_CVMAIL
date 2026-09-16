#!/usr/bin/env bash
#SBATCH --job-name=tiger_cs_t2_p2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --array=0-4%2
#SBATCH --output=full_version/center_stratified_32_8/logs/task2_p2_%A_%a.log
#SBATCH --error=full_version/center_stratified_32_8/logs/task2_p2_%A_%a.log
set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python full_version/task2_fine_31cls/scripts/train_full.py \
  --fold "${SLURM_ARRAY_TASK_ID}" \
  --config full_version/center_stratified_32_8/configs/task2_fine_p2_hr_best.yaml \
  --split-manifest full_version/center_stratified_32_8/splits/case_folds.json \
  --output-root full_version/center_stratified_32_8/artifacts

