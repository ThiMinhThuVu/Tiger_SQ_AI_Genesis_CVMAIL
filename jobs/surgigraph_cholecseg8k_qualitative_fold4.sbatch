#!/usr/bin/env bash
#SBATCH --job-name=sggf-qual-f4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/surgigraph_cholecseg8k_qualitative_fold4_%j.log
#SBATCH --error=logs/surgigraph_cholecseg8k_qualitative_fold4_%j.log

set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate
export MPLCONFIGDIR=/tmp/matplotlib-${SLURM_JOB_ID}

python scripts/generate_surgigraph_cholecseg8k_qualitative.py \
  --config configs/surgigraph_q_a3_r2_gf_cholecseg8k.yaml \
  --data-root cross_data/data/CholecSeg8k \
  --output-root artifacts/cholecseg8k_surgigraph \
  --fold 4
