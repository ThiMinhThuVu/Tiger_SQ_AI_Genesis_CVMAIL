#!/usr/bin/env bash
#SBATCH --job-name=tiger-sam2t-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/sam2_tiny_smoke_%j.log
#SBATCH --error=logs/sam2_tiny_smoke_%j.log

set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate
nvidia-smi
python scripts/train_sam.py \
  --model sam2_hiera_tiny --fold 0 --n-folds 5 --epochs 1 \
  --height 512 --width 896 --batch-size 1 --gradient-accumulation 4 \
  --gradient-checkpointing --finetune partial --mixed-precision bf16 \
  --num-workers 4 --seed 2026 --run-kind smoke \
  --output-dir artifacts/new_baselines/sam2_hiera_tiny_partial/engineering/smoke/fold_0
python scripts/predict_sam.py \
  --checkpoint artifacts/new_baselines/sam2_hiera_tiny_partial/engineering/smoke/fold_0/best.pt \
  --fold 0 \
  --output-dir artifacts/new_baselines/sam2_hiera_tiny_partial/engineering/smoke/fold_0/predictions
python scripts/validate_predictions.py \
  --prediction-dir artifacts/new_baselines/sam2_hiera_tiny_partial/engineering/smoke/fold_0/predictions \
  --fold 0 --run-official --official-fixture \
  --output artifacts/new_baselines/sam2_hiera_tiny_partial/engineering/smoke/fold_0/prediction_validation.json
