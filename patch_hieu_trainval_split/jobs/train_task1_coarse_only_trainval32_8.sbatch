#!/usr/bin/env bash
#SBATCH --job-name=t1_coarse_tv32_8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-4%2
#SBATCH --output=full_version/task1_merged_16cls/logs/train_tv32_8_%A_%a.log
#SBATCH --error=full_version/task1_merged_16cls/logs/train_tv32_8_%A_%a.log

# Draft job for the 32-train/8-val split (no held-out local test tier).
# Same prerequisites as ../jobs/train_task2_fine31_trainval32_8.sbatch, plus
# train_coarse_only.py must accept the v2 manifest version (see
# ../train_coarse_only.py.patch).

set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate
python full_version/task1_merged_16cls/scripts/train_coarse_only.py \
  --mode train --fold "${SLURM_ARRAY_TASK_ID}" \
  --config full_version/task1_merged_16cls/configs/coarse_only_swin_small_trainval32_8_seed2026.yaml \
  --split-manifest full_version/task2_fine_31cls/splits/case_folds_v2_trainval.json \
  --output-root full_version/task1_merged_16cls/artifacts/seed_2026_trainval32_8
