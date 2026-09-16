#!/usr/bin/env bash
#SBATCH --job-name=dsml-sf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-7%1
#SBATCH --output=logs/dsml_phase0_seeds_folds_a0_b1_%A_%a.log
#SBATCH --error=logs/dsml_phase0_seeds_folds_a0_b1_%A_%a.log

set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Independent of job 265 (A0-matched seed2026 fold0, running standalone on the
# other GPU). This array is capped at %1 so this array + job 265 together stay
# at exactly 2 concurrent GPUs (see reports/A_STAR_2027_NOVELTY_SCAN_BROADENED.md
# section 8.1).
configs=(
  "mask2former_swin_small_c4_a0_nocontext_bs4_task1_seed2027"
  "mask2former_swin_small_c4_a0_nocontext_bs4_task1_seed2028"
  "mask2former_swin_small_b1_recall_reweight_fold0_seed2027"
  "mask2former_swin_small_b1_recall_reweight_fold0_seed2028"
  "mask2former_swin_small_c4_a0_nocontext_bs4_task1_fold1"
  "mask2former_swin_small_c4_a0_nocontext_bs4_task1_fold2"
  "mask2former_swin_small_b1_recall_reweight_fold1"
  "mask2former_swin_small_b1_recall_reweight_fold2"
)
target_folds=(0 0 0 0 1 2 1 2)
experiment_id="${configs[$SLURM_ARRAY_TASK_ID]}"
target_fold="${target_folds[$SLURM_ARRAY_TASK_ID]}"
config="configs/${experiment_id}.yaml"
output="artifacts/dsml_phase0/${experiment_id}/fold_${target_fold}"

if [[ -e "${output}/best.pt" || -e "${output}/completion.json" ]]; then
  echo "Refusing to overwrite completed result ${output}" >&2
  exit 2
fi

echo "Running ${experiment_id} (array task ${SLURM_ARRAY_TASK_ID}) on fold ${target_fold}"
python scripts/train_mask2former_improved.py \
  --config "${config}" \
  --fold "${target_fold}" \
  --output-root artifacts/dsml_phase0
