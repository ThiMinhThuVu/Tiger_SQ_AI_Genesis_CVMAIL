#!/usr/bin/env bash
#SBATCH --job-name=c3-ot-t1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=36:00:00
#SBATCH --array=0-4%2
#SBATCH --output=logs/mask2former_c3_ot_task1_%A_%a.log
#SBATCH --error=logs/mask2former_c3_ot_task1_%A_%a.log

set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

output="artifacts/mask2former_research/mask2former_swin_small_c3_ot_task1/fold_${SLURM_ARRAY_TASK_ID}"
if [[ -e "${output}" ]]; then
  echo "Refusing to overwrite existing ${output}" >&2
  exit 2
fi

python scripts/train_mask2former_improved.py \
  --config configs/mask2former_swin_small_c3_ot_task1.yaml \
  --fold "${SLURM_ARRAY_TASK_ID}" \
  --output-root artifacts/mask2former_research
