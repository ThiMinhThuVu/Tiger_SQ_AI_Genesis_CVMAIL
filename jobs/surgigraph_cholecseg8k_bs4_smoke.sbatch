#!/usr/bin/env bash
#SBATCH --job-name=sggf-bs4-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/surgigraph_cholecseg8k_bs4_smoke_%j.log
#SBATCH --error=logs/surgigraph_cholecseg8k_bs4_smoke_%j.log

set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/train_surgigraph_cholecseg8k.py \
  --config configs/surgigraph_q_a3_r2_gf_cholecseg8k_bs4_smoke.yaml \
  --data-root cross_data/data/CholecSeg8k \
  --output-root artifacts/cholecseg8k_surgigraph_smoke \
  --fold 0 \
  --smoke
