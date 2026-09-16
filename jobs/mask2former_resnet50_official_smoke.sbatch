#!/usr/bin/env bash
#SBATCH --job-name=m2f-r50-official-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/mask2former_resnet50_official_smoke_%j.log
#SBATCH --error=logs/mask2former_resnet50_official_smoke_%j.log
set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate
python scripts/train_mask2former.py --config configs/mask2former_resnet50_official.yaml --smoke
