#!/usr/bin/env bash
#SBATCH --job-name=sggf-cholec-test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-4%2
#SBATCH --output=logs/surgigraph_q_a3_r2_gf_cholecseg8k_test_%A_%a.log
#SBATCH --error=logs/surgigraph_q_a3_r2_gf_cholecseg8k_test_%A_%a.log

set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate

python scripts/train_surgigraph_cholecseg8k.py \
  --config configs/surgigraph_q_a3_r2_gf_cholecseg8k.yaml \
  --data-root cross_data/data/CholecSeg8k \
  --output-root artifacts/cholecseg8k_surgigraph \
  --fold "${SLURM_ARRAY_TASK_ID}" \
  --split test
