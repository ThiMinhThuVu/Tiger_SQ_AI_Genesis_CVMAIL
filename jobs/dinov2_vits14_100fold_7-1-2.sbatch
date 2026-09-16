#!/usr/bin/env bash
#SBATCH --job-name=dino-vits14-cv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/dinov2_vits14_cv_%j.log
#SBATCH --error=logs/dinov2_vits14_cv_%j.log
set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate
python scripts/train_dinov2.py --config configs/dinov2_vits14.yaml --run-all-folds
