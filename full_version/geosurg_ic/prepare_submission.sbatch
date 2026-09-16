#!/usr/bin/env bash
#SBATCH --job-name=tiger_geosurg_package
#SBATCH --partition=gpu
#SBATCH --nodelist=vishc-server-2
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=full_version/geosurg_ic/prepare_submission_%j.log
#SBATCH --error=full_version/geosurg_ic/prepare_submission_%j.log
set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
export CUDA_VISIBLE_DEVICES=GPU-f23bb72c-7b5a-84ca-623e-9d2b33d031e5
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
timeout 3500 .venv/bin/python -u full_version/geosurg_ic/prepare_submission.py
