#!/usr/bin/env bash
#SBATCH --job-name=nnunet-test-quant
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/nnunet_test_quant_%j.log
#SBATCH --error=logs/nnunet_test_quant_%j.log
set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate
export nnUNet_raw="$PWD/artifacts/nnunet_raw"
export nnUNet_preprocessed="$PWD/artifacts/nnunet_preprocessed"
export nnUNet_results="$PWD/artifacts/nnunet_results"
export nnUNet_extTrainer="$PWD/jobs/nnunet_external_trainer"
python scripts/run_nnunet_test.py
