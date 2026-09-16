#!/usr/bin/env bash
#SBATCH --job-name=safegraph-mpkg-v2
#SBATCH --partition=gpu
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=logs/safe_graph_multipair_kg_v2_%j.log
#SBATCH --error=logs/safe_graph_multipair_kg_v2_%j.log

set -euo pipefail

cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate

python -m pytest -q \
  tests/test_safe_graph_multipair_kg_v2.py \
  tests/test_build_safe_graph_knowledge.py \
  tests/test_safe_graph_error_corpus.py

python scripts/evaluate_safe_graph_multipair_kg_v2.py \
  --error-corpus artifacts/safe_graph_error_corpus/improved_v2_validation_error_v1_r1 \
  --oof-root artifacts/safe_graph_oof/improved_v2_validation_v1 \
  --kg-root artifacts/safe_graph_knowledge/kg_v1_20260808 \
  --output-root artifacts/safe_graph_verifier/multipair_kg_v2
