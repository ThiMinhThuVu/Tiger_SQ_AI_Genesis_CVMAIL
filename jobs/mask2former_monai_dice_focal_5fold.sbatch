#!/usr/bin/env bash
#SBATCH --job-name=m2f-monai
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/mask2former_monai_%A_%a.log
#SBATCH --error=logs/mask2former_monai_%A_%a.log
set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate
nvidia-smi
python -c 'import monai; assert monai.__version__ == "1.5.2", monai.__version__'

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    python scripts/train_convnextv2_mask2former.py \
      --config configs/convnextv2_tiny_mask2former_monai_dice_focal.yaml \
      --run-all-folds
    ;;
  1)
    python scripts/train_mask2former.py \
      --config configs/mask2former_swin_tiny_monai_dice_focal.yaml \
      --run-all-folds
    ;;
  *)
    echo "Unexpected SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac
