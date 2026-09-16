#!/usr/bin/env bash
#SBATCH --job-name=m2f-swin-tiny-cv100
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/mask2former_swin_tiny_100fold_7-1-2_%j.log
#SBATCH --error=logs/mask2former_swin_tiny_100fold_7-1-2_%j.log
set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate
python scripts/train_mask2former.py --config configs/mask2former_swin_tiny.yaml --run-all-folds
