#!/usr/bin/env bash
#SBATCH --job-name=m2f-qual-test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/mask2former_qualitative_%j.log
#SBATCH --error=logs/mask2former_qualitative_%j.log
set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate
for fold in 0 1 2 3 4; do python scripts/generate_mask2former_qualitative.py --config configs/mask2former_swin_tiny.yaml --checkpoint artifacts/mask2former/mask2former_swin_tiny/fold_${fold}/best.pt --fold ${fold} --split test --output-dir reports/qualitative/mask2former_swin_tiny/test/fold_${fold}; done
