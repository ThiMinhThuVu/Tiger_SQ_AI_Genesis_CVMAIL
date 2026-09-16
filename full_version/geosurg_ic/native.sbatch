#!/usr/bin/env bash
#SBATCH --job-name=tiger_geosurg_native
#SBATCH --partition=gpu
#SBATCH --nodelist=vishc-server-2
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=full_version/geosurg_ic/artifacts/pilot_v1/native_%j.log
#SBATCH --error=full_version/geosurg_ic/artifacts/pilot_v1/native_%j.log
set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
export CUDA_VISIBLE_DEVICES=GPU-f23bb72c-7b5a-84ca-623e-9d2b33d031e5
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1
timeout 7000 .venv/bin/python -u full_version/geosurg_ic/evaluate_native.py \
    --pilot full_version/geosurg_ic/artifacts/pilot_v1 --fold 0
