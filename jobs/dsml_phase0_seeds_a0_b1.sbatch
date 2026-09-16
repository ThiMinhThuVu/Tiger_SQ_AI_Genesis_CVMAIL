#!/usr/bin/env bash
#SBATCH --job-name=dsml-seeds
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-3%2
#SBATCH --output=logs/dsml_phase0_seeds_a0_b1_%A_%a.log
#SBATCH --error=logs/dsml_phase0_seeds_a0_b1_%A_%a.log

set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Seed replicates for the two live arms of the Phase-0 comparison (see
# reports/A_STAR_2027_NOVELTY_SCAN_BROADENED.md section 8.1): does B1's
# per-class reweight beat A0, or is the whole B1-B4 spread environment noise
# (torch 2.11 here vs the repo-pinned torch 2.7.1 that A0/B1-B4's seed-2026
# runs actually used). B2/B3/B4 excluded on purpose -- their sink-margin term
# was measured near-zero at convergence, so more seeds of an inactive term
# add nothing.
configs=(
  "mask2former_swin_small_c4_a0_nocontext_bs4_task1_seed2027"
  "mask2former_swin_small_c4_a0_nocontext_bs4_task1_seed2028"
  "mask2former_swin_small_b1_recall_reweight_fold0_seed2027"
  "mask2former_swin_small_b1_recall_reweight_fold0_seed2028"
)
experiment_id="${configs[$SLURM_ARRAY_TASK_ID]}"
config="configs/${experiment_id}.yaml"
output="artifacts/dsml_phase0/${experiment_id}/fold_0"

if [[ -e "${output}/best.pt" || -e "${output}/completion.json" ]]; then
  echo "Refusing to overwrite completed result ${output}" >&2
  exit 2
fi

echo "Running ${experiment_id} (array task ${SLURM_ARRAY_TASK_ID}) with config ${config}"
python scripts/train_mask2former_improved.py \
  --config "${config}" \
  --fold 0 \
  --output-root artifacts/dsml_phase0
