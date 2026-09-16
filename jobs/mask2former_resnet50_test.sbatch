#!/usr/bin/env bash
#SBATCH --job-name=m2f-r50-test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/mask2former_resnet50_test_%j.log
#SBATCH --error=logs/mask2former_resnet50_test_%j.log
set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate
python scripts/infer_mask2former_test.py --config configs/mask2former_resnet50.yaml --model-dir artifacts/mask2former/mask2former_resnet50 --output artifacts/mask2former/mask2former_resnet50/test_quantitative.json
