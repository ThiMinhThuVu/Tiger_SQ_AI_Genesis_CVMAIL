#!/usr/bin/env bash
#SBATCH --job-name=safegraph-kg-v1
#SBATCH --partition=gpu
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/safe_graph_kg_v1_%j.log
#SBATCH --error=logs/safe_graph_kg_v1_%j.log

set -euo pipefail

cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate

python -m pytest -q tests/test_build_safe_graph_knowledge.py

python scripts/build_safe_graph_knowledge.py \
  --data-root data \
  --seed knowledge_graph/clinical_station_anatomy_seed_v1.json \
  --output-root artifacts/safe_graph_knowledge/kg_v1_20260808 \
  --all-folds \
  --include-all-data-descriptive \
  --relation-max-side 512 \
  --touch-pixels 2
