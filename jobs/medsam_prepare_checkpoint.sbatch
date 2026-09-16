#!/usr/bin/env bash
#SBATCH --job-name=medsam-ckpt
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/medsam_checkpoint_%j.log
#SBATCH --error=logs/medsam_checkpoint_%j.log
set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate
mkdir -p checkpoints/medsam
if [ ! -f checkpoints/medsam/medsam_vit_b.pth ] || [ "$(stat -c%s checkpoints/medsam/medsam_vit_b.pth)" -lt 300000000 ]; then
  gdown 1UAmWL88roYR7wKlnApw5Bcuzf2iQgk6_ -O checkpoints/medsam/medsam_vit_b.pth
fi
test "$(stat -c%s checkpoints/medsam/medsam_vit_b.pth)" -gt 300000000
