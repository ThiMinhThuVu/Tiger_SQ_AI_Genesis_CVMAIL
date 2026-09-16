#!/usr/bin/env bash
#SBATCH --job-name=safegraph-vroi
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/safe_graph_visual_roi_%j.log
#SBATCH --error=logs/safe_graph_visual_roi_%j.log

set -euo pipefail

cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate

python -m pytest -q \
  tests/test_safe_graph_visual_roi_features.py \
  tests/test_safe_graph_p2_dense_cf005.py

python scripts/export_safe_graph_visual_roi_features.py \
  --pair-cohort artifacts/safe_graph_verifier/p2_dense_cf005_v1/dense_pair_cohort.csv \
  --reference-oof-root artifacts/safe_graph_oof/improved_v2_validation_p2_dense_cf005_v1 \
  --dinov2-root artifacts/new_baselines/dinov2_vits14 \
  --data-root data \
  --output-root artifacts/safe_graph_verifier/visual_roi_cf005_v1
