#!/usr/bin/env bash
#SBATCH --job-name=safegraph-oof-v1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/safe_graph_oof_v1_%j.log
#SBATCH --error=logs/safe_graph_oof_v1_%j.log

set -euo pipefail

cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate

python -m pytest -q tests/test_safe_graph_oof_export.py

python scripts/export_safe_graph_oof.py \
  --config configs/mask2former_swin_small_clinical_hier_presence_v2.yaml \
  --model-dir artifacts/mask2former_improved/mask2former_swin_small_clinical_hier_presence_v2 \
  --data-root data \
  --output-root artifacts/safe_graph_oof/improved_v2_validation_v1 \
  --folds 0 1 2 3 4 \
  --top-k 3
