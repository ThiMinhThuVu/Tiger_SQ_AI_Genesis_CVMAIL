#!/usr/bin/env bash
#SBATCH --job-name=tiger_p0_native
#SBATCH --partition=gpu
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=full_version/task2_fine_31cls/logs/p0_finalize_%j.log
#SBATCH --error=full_version/task2_fine_31cls/logs/p0_finalize_%j.log

set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate

python full_version/task2_fine_31cls/scripts/p0_baseline_lock.py \
  --stage finalize \
  --metric-workers "${SLURM_CPUS_PER_TASK}" \
  --config full_version/task2_fine_31cls/configs/full40_fine31_c4_a0_seed2026.yaml \
  --split-manifest full_version/task2_fine_31cls/splits/case_folds_v1.json \
  --model-dir full_version/task2_fine_31cls/artifacts/seed_2026/full40_fine31_c4_a0_seed2026 \
  --output-dir full_version/task2_fine_31cls/artifacts/p0_baseline_lock_v2 \
  --data-root data
