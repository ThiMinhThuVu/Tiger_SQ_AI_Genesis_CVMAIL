#!/usr/bin/env bash
#SBATCH --job-name=cv2base-m2f-cv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=120:00:00
#SBATCH --output=logs/convnextv2_base_mask2former_%j.log
#SBATCH --error=logs/convnextv2_base_mask2former_%j.log
set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate
nvidia-smi
python scripts/train_convnextv2_mask2former.py --config configs/convnextv2_base_mask2former.yaml --run-all-folds
