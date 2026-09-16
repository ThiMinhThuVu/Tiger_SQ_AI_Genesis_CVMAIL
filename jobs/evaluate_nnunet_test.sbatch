#!/usr/bin/env bash
#SBATCH --job-name=nnunet-eval-quant
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/nnunet_eval_quant_%j.log
#SBATCH --error=logs/nnunet_eval_quant_%j.log
set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate
python scripts/evaluate_nnunet_test.py \
  --data-root data \
  --prediction-root artifacts/nnunet_results/Dataset501_TIGER/nnUNetTrainer_TIGER100__nnUNetPlans__2d \
  --output artifacts/nnunet_results/Dataset501_TIGER/nnUNetTrainer_TIGER100__nnUNetPlans__2d/test_quantitative.json
