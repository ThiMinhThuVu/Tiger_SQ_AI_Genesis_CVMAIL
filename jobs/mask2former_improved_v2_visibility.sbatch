#!/usr/bin/env bash
#SBATCH --job-name=m2f-v2-vis
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/mask2former_improved_v2_visibility_%j.log
#SBATCH --error=logs/mask2former_improved_v2_visibility_%j.log
set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate
python scripts/train_mask2former_visibility.py \
  --config configs/mask2former_swin_small_clinical_hier_presence_v2.yaml \
  --model-dir artifacts/mask2former_improved/mask2former_swin_small_clinical_hier_presence_v2 \
  --segmentation-results artifacts/mask2former_improved/mask2former_swin_small_clinical_hier_presence_v2/test_quantitative_r1.json \
  --output-root artifacts/mask2former_improved/mask2former_swin_small_clinical_hier_presence_v2_task3
