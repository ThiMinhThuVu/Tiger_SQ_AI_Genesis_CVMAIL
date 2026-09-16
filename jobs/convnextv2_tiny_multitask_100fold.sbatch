#!/usr/bin/env bash
#SBATCH --job-name=convnextv2-tiny-cv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=logs/convnextv2_tiny_%j.log
#SBATCH --error=logs/convnextv2_tiny_%j.log
set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate
nvidia-smi
python scripts/train_convnextv2.py --config configs/convnextv2_tiny_multitask.yaml --run-all-folds
python scripts/infer_convnextv2_test.py --config configs/convnextv2_tiny_multitask.yaml --model-dir artifacts/new_baselines/convnextv2_tiny_multitask --output artifacts/new_baselines/convnextv2_tiny_multitask/test_quantitative.json
for f in 0 1 2 3 4; do
  python scripts/generate_convnextv2_qualitative.py --config configs/convnextv2_tiny_multitask.yaml --checkpoint artifacts/new_baselines/convnextv2_tiny_multitask/fold_${f}/best.pt --fold ${f} --output-dir artifacts/new_baselines/convnextv2_tiny_multitask/fold_${f}/qualitative
done
