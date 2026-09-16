#!/usr/bin/env bash
#SBATCH --job-name=grasp-cholec-test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-4%2
#SBATCH --output=logs/grasp_cholecseg8k_test_%A_%a.log
#SBATCH --error=logs/grasp_cholecseg8k_test_%A_%a.log

set -euo pipefail
ROOT=/mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
TAPIS="$ROOT/test_sota/GraSP/TAPIS"
FOLD="${SLURM_ARRAY_TASK_ID}"

cd "$ROOT"
source .venv/bin/activate
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"

cd "$TAPIS/region_proposals"
python train_net.py --num-gpus 1 --eval-only \
  --config-file configs/cholecseg8k/GraSP_R50_CholecSeg8k.yaml \
  DATASETS.TEST "('cholecseg8k_test',)" \
  MODEL.WEIGHTS "$ROOT/outputs/grasp_cholecseg8k/fold${FOLD}/model_final.pth" \
  OUTPUT_DIR "$ROOT/outputs/grasp_cholecseg8k/test_fold${FOLD}"
