#!/usr/bin/env bash
#SBATCH --job-name=tiger-sam2t-stability
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/sam2_tiny_stability_%j.log
#SBATCH --error=logs/sam2_tiny_stability_%j.log

set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate
nvidia-smi
output=artifacts/new_baselines/sam2_hiera_tiny_partial/engineering/stability/fold_0
python scripts/train_sam.py \
  --model sam2_hiera_tiny --fold 0 --n-folds 5 --epochs 10 \
  --height 512 --width 896 --batch-size 1 --gradient-accumulation 4 \
  --gradient-checkpointing --finetune partial --mixed-precision bf16 \
  --num-workers 4 --seed 2026 --run-kind stability --stop-after-epoch 5 \
  --output-dir "${output}"
python scripts/train_sam.py \
  --model sam2_hiera_tiny --fold 0 --n-folds 5 --epochs 10 \
  --height 512 --width 896 --batch-size 1 --gradient-accumulation 4 \
  --gradient-checkpointing --finetune partial --mixed-precision bf16 \
  --num-workers 4 --seed 2026 --run-kind stability --resume \
  --output-dir "${output}"
python scripts/predict_sam.py \
  --checkpoint "${output}/best.pt" --fold 0 --output-dir "${output}/predictions"
python scripts/validate_predictions.py \
  --prediction-dir "${output}/predictions" --fold 0 --run-official --official-fixture \
  --output "${output}/prediction_validation.json"
python scripts/check_sam_stability.py
