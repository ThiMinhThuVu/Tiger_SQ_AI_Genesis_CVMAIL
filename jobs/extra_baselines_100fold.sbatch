#!/usr/bin/env bash
#SBATCH --job-name=extra-baselines-cv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --output=logs/extra_baselines_%j.log
#SBATCH --error=logs/extra_baselines_%j.log
set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate
nvidia-smi
python scripts/train_deeplabv3.py --config configs/deeplabv3_resnet50_multitask.yaml --run-all-folds
# ConvNeXt-V2 Base + Mask2Former is launched by the dedicated runner once the
# transformers Mask2Former backend is available on the GPU image.
