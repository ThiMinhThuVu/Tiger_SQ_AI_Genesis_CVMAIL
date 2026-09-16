#!/usr/bin/env bash
#SBATCH --job-name=m2f-clinical-b0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=logs/mask2former_swin_small_clinical_only_b0_%j.log
#SBATCH --error=logs/mask2former_swin_small_clinical_only_b0_%j.log
set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate
python scripts/train_mask2former_improved.py \
  --config configs/mask2former_swin_small_clinical_only_b0.yaml \
  --run-all-folds --output-root artifacts/mask2former_improved
