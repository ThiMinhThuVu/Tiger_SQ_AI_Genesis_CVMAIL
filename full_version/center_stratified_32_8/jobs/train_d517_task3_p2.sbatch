#!/usr/bin/env bash
#SBATCH --job-name=tiger_d517_t3_p2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=full_version/center_stratified_32_8/logs/d517_task3_p2_%j.log
#SBATCH --error=full_version/center_stratified_32_8/logs/d517_task3_p2_%j.log
set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
gpu=3
echo "Pinned physical GPU: ${gpu}"
CUDA_VISIBLE_DEVICES="${gpu}" python full_version/center_stratified_32_8/scripts/train_task3.py \
  --config full_version/center_stratified_32_8/configs/d517_task2_fine_p2.yaml \
  --manifest full_version/center_stratified_32_8/splits/case_folds_d517.json \
  --model-dir full_version/center_stratified_32_8/artifacts/center_strat32x8_d517_task2_fine_p2_hr_improved_v2 \
  --output-root full_version/center_stratified_32_8/artifacts/d517_task3_visibility_p2
