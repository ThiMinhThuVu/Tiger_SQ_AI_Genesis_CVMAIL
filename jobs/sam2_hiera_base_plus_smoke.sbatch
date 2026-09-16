#!/usr/bin/env bash
#SBATCH --job-name=tiger-sam2-base-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/sam2_hiera_base_plus_smoke_%j.log
#SBATCH --error=logs/sam2_hiera_base_plus_smoke_%j.log

set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate
output=artifacts/new_baselines/sam2_hiera_base_plus_partial/engineering/smoke/fold_0
python scripts/train_sam.py --model sam2_hiera_base_plus --fold 0 --n-folds 5 \
  --epochs 1 --height 512 --width 896 --batch-size 1 --gradient-accumulation 4 \
  --gradient-checkpointing --finetune partial --mixed-precision bf16 \
  --num-workers 4 --seed 2026 --run-kind smoke \
  --output-dir "${output}"
python scripts/predict_sam.py --checkpoint "${output}/best.pt" --fold 0 \
  --output-dir "${output}/predictions"
python scripts/validate_predictions.py --prediction-dir "${output}/predictions" \
  --fold 0 --output "${output}/prediction_validation.json"
python scripts/validate_predictions.py --prediction-dir "${output}/predictions" \
  --fold 0 --run-official --official-fixture \
  --output "${output}/official_fixture_validation.json"
