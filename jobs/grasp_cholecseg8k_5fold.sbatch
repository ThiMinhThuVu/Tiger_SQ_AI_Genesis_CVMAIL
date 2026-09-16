#!/usr/bin/env bash
#SBATCH --job-name=grasp-cholec-cv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --array=0-4%2
#SBATCH --output=logs/grasp_cholecseg8k_%A_%a.log
#SBATCH --error=logs/grasp_cholecseg8k_%A_%a.log

set -euo pipefail
ROOT=/mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
TAPIS="$ROOT/test_sota/GraSP/TAPIS"
DATA="$ROOT/cross_data/data/CholecSeg8k"
FOLD="${SLURM_ARRAY_TASK_ID}"

cd "$ROOT"
source .venv/bin/activate
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
python "$TAPIS/tools/prepare_cholecseg8k.py" --data-root "$DATA"

cd "$TAPIS/region_proposals"
python train_net.py --num-gpus 1 \
  --config-file configs/cholecseg8k/GraSP_R50_CholecSeg8k.yaml \
  DATASETS.TRAIN "('cholecseg8k_train_fold${FOLD}',)" \
  DATASETS.TEST "('cholecseg8k_val_fold${FOLD}',)" \
  OUTPUT_DIR "$ROOT/outputs/grasp_cholecseg8k/fold${FOLD}"
