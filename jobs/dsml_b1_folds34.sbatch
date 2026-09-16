#!/usr/bin/env bash
#SBATCH --job-name=dsml-b1-f34
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-1%2
#SBATCH --output=logs/dsml_b1_folds34_%A_%a.log
#SBATCH --error=logs/dsml_b1_folds34_%A_%a.log

set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

configs=(
  "mask2former_swin_small_b1_recall_reweight_fold3"
  "mask2former_swin_small_b1_recall_reweight_fold4"
)
target_folds=(3 4)
experiment_id="${configs[$SLURM_ARRAY_TASK_ID]}"
target_fold="${target_folds[$SLURM_ARRAY_TASK_ID]}"
output_root="artifacts/dsml_phase0_b1_folds34"
output="${output_root}/${experiment_id}/fold_${target_fold}"

if [[ -e "${output}/best.pt" || -e "${output}/completion.json" ]]; then
  echo "Refusing to overwrite completed result ${output}" >&2
  exit 2
fi

python scripts/train_mask2former_improved.py \
  --config "configs/${experiment_id}.yaml" \
  --fold "${target_fold}" \
  --output-root "${output_root}"
