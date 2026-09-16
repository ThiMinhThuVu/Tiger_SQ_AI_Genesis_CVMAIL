#!/usr/bin/env bash
#SBATCH --job-name=medsam-vitb-cv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/medsam_vit_b_cv_%j.log
#SBATCH --error=logs/medsam_vit_b_cv_%j.log
set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate
test -f checkpoints/medsam/medsam_vit_b.pth
python scripts/train_medsam.py --config configs/medsam_vit_b.yaml --run-all-folds
