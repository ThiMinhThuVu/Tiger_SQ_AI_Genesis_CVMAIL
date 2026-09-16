#!/usr/bin/env bash
#SBATCH --job-name=tiger_geosurg_ic
#SBATCH --partition=gpu
#SBATCH --nodelist=vishc-server-2
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=full_version/geosurg_ic/artifacts/pilot_v1/controller_%j.log
#SBATCH --error=full_version/geosurg_ic/artifacts/pilot_v1/controller_%j.log
set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
# This cluster's GRES does not reliably hide physical devices. Explicit UUID
# pinning keeps every subprocess on exactly the SAME ONE physical GPU.
export CUDA_VISIBLE_DEVICES=GPU-f23bb72c-7b5a-84ca-623e-9d2b33d031e5
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for arm in aug geo gt shuffle; do
    echo "Starting arm=${arm} date=$(date -u +%FT%TZ)"
    timeout 5400 .venv/bin/python -u full_version/geosurg_ic/run_pilot.py \
        --arm "${arm}" --fold 0 --steps 192 --ic-every 2 --ic-weight 0.02 \
        --output full_version/geosurg_ic/artifacts/pilot_v1 \
        >"full_version/geosurg_ic/artifacts/pilot_v1/${arm}.log" 2>&1
    echo "Completed arm=${arm} date=$(date -u +%FT%TZ)"
done
