#!/usr/bin/env bash
#SBATCH --job-name=nnunet-prepare
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/nnunet_prepare_%j.log
#SBATCH --error=logs/nnunet_prepare_%j.log
set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate
python scripts/prepare_nnunet_dataset.py
