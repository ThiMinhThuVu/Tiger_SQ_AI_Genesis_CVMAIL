#!/usr/bin/env bash
#SBATCH --job-name=tiger_geosurg_selected
#SBATCH --partition=gpu
#SBATCH --nodelist=vishc-server-2
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=full_version/geosurg_ic/selected_5fold_%j.log
#SBATCH --error=full_version/geosurg_ic/selected_5fold_%j.log
set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
export CUDA_VISIBLE_DEVICES=GPU-f23bb72c-7b5a-84ca-623e-9d2b33d031e5
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Fold 0 is the completed pilot checkpoint. The remaining folds use precisely
# the same recipe, initialized from each corresponding D517 P2 checkpoint.
for arm in aug geo; do
    for fold in 1 2 3 4; do
        echo "Starting arm=${arm} fold=${fold} date=$(date -u +%FT%TZ)"
        timeout 1200 .venv/bin/python -u full_version/geosurg_ic/run_pilot.py \
            --arm "${arm}" --fold "${fold}" --steps 192 --ic-every 2 --ic-weight 0.02 \
            --output full_version/geosurg_ic/artifacts/pilot_v1 \
            >"full_version/geosurg_ic/artifacts/pilot_v1/${arm}_fold${fold}.log" 2>&1
        echo "Completed arm=${arm} fold=${fold} date=$(date -u +%FT%TZ)"
    done
done
