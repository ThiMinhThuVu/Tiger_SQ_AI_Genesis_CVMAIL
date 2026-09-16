#!/usr/bin/env bash
#SBATCH --job-name=deeplab-test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/deeplab_test_%j.log
#SBATCH --error=logs/deeplab_test_%j.log
set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate
python scripts/evaluate_extra_multitask_test.py --model deeplabv3 --config configs/deeplabv3_resnet50_multitask.yaml --model-dir artifacts/new_baselines/deeplabv3_resnet50_multitask --output artifacts/new_baselines/deeplabv3_resnet50_multitask/test_quantitative.json
