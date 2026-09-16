#!/usr/bin/env bash
#SBATCH --job-name=tiger-sam2-small-f0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/sam2_hiera_small_fold0_primary_80_%j.log
#SBATCH --error=logs/sam2_hiera_small_fold0_primary_80_%j.log

set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate
output=artifacts/new_baselines/sam2_hiera_small_partial/fold_0
python scripts/update_experiment_registry.py sam2_hiera_small_partial \
  --status running --last-completed-phase fold_0_primary_80_epoch_started
python scripts/train_sam.py --model sam2_hiera_small --fold 0 --n-folds 5 \
  --epochs 80 --height 512 --width 896 --batch-size 1 --gradient-accumulation 4 \
  --gradient-checkpointing --finetune partial --mixed-precision bf16 \
  --num-workers 4 --seed 2026 --run-kind primary --output-dir "${output}"
python scripts/predict_sam.py --checkpoint "${output}/best.pt" --fold 0 \
  --output-dir "${output}/predictions"
python scripts/validate_predictions.py --prediction-dir "${output}/predictions" \
  --fold 0 --output "${output}/prediction_validation.json"
python scripts/validate_predictions.py --prediction-dir "${output}/predictions" \
  --fold 0 --run-official --official-fixture \
  --output "${output}/official_fixture_validation.json"
python scripts/prepare_official_native_input.py --prediction-dir "${output}/predictions" \
  --fold 0 --destination "${output}/official_native_input"
python scripts/run_native_official_fast.py \
  --gt "${output}/official_native_input/gt" \
  --pred "${output}/official_native_input/predictions" \
  --out "${output}/official_native_fast_report.md" --figures-dir none
python scripts/generate_sam_qualitative.py --prediction-dir "${output}/predictions" \
  --fold 0 --output-dir "${output}/qualitative"
python scripts/update_experiment_registry.py sam2_hiera_small_partial \
  --status running --last-completed-phase fold_0_primary_80_epoch_completed \
  --completed-folds '[0]'
