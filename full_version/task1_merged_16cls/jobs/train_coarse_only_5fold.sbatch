#!/usr/bin/env bash
#SBATCH --job-name=t1_coarse_full40
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-4%2
#SBATCH --output=full_version/task1_merged_16cls/logs/train_%A_%a.log
#SBATCH --error=full_version/task1_merged_16cls/logs/train_%A_%a.log
set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate
python full_version/task1_merged_16cls/scripts/train_coarse_only.py \
  --mode train --fold "${SLURM_ARRAY_TASK_ID}" \
  --config full_version/task1_merged_16cls/configs/coarse_only_swin_small_seed2026.yaml \
  --split-manifest full_version/task2_fine_31cls/splits/case_folds_v1.json \
  --output-root full_version/task1_merged_16cls/artifacts/seed_2026
