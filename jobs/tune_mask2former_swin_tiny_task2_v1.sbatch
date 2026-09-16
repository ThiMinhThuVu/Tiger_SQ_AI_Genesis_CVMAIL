#!/usr/bin/env bash
#SBATCH --job-name=m2f-swin-t2-v1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/mask2former_swin_tiny_task2_tuned_v1_%j.log
#SBATCH --error=logs/mask2former_swin_tiny_task2_tuned_v1_%j.log
set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate
python scripts/tune_mask2former_task2.py \
  --config configs/mask2former_swin_tiny.yaml \
  --model-dir artifacts/mask2former/mask2former_swin_tiny \
  --output-dir artifacts/mask2former/mask2former_swin_tiny_task2_tuned_v1
