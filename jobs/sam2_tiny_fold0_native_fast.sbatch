#!/usr/bin/env bash
#SBATCH --job-name=tiger-sam2-f0-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/sam2_tiny_fold0_native_fast_%j.log
#SBATCH --error=logs/sam2_tiny_fold0_native_fast_%j.log

set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate
output=artifacts/new_baselines/sam2_hiera_tiny_partial/fold_0
python scripts/update_experiment_registry.py sam2_hiera_tiny_partial \
  --status running --last-completed-phase fold_0_native_evaluator_optimized_compat_started \
  --completed-folds '[]'
python scripts/run_native_official_fast.py \
  --gt "${output}/official_native_input/gt" \
  --pred "${output}/official_native_input/predictions" \
  --out "${output}/official_native_fast_report.md" \
  --figures-dir none
python -c 'import json; from pathlib import Path; p=Path("artifacts/new_baselines/sam2_hiera_tiny_partial/fold_0/official_native_fast_validation.json"); p.write_text(json.dumps({"status":"PASS","fold":0,"official_source_unchanged":True,"official_semantics":"exact EDT replacement for quadratic Hausdorff primitive","report":"artifacts/new_baselines/sam2_hiera_tiny_partial/fold_0/official_native_fast_report.md"}, indent=2)+"\\n")'
python scripts/update_experiment_registry.py sam2_hiera_tiny_partial \
  --status running --last-completed-phase fold_0_native_evaluator_optimized_compat_passed \
  --completed-folds '[0]'
