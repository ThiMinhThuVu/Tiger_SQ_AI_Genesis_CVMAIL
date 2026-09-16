#!/usr/bin/env bash
#SBATCH --job-name=tiger_geosurg_submit
#SBATCH --partition=gpu
#SBATCH --nodelist=vishc-server-2
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:30:00
#SBATCH --output=full_version/geosurg_ic/upload_selected_%j.log
#SBATCH --error=full_version/geosurg_ic/upload_selected_%j.log
set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
export CUDA_VISIBLE_DEVICES=''
timeout 5300 /home/thu.vtm/miniconda3/bin/python -u full_version/geosurg_ic/upload_selected.py \
    --plan full_version/geosurg_ic/submission/upload_plan.json
