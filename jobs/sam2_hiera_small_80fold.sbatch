#!/usr/bin/env bash
#SBATCH --job-name=tiger-sam2-small-cv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/sam2_hiera_small_80fold_%j.log
#SBATCH --error=logs/sam2_hiera_small_80fold_%j.log
set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate
bash scripts/run_sam2_variant_80fold_fast.sh sam2_hiera_small sam2_hiera_small_partial
