#!/usr/bin/env bash
#SBATCH --job-name=dsml-p0-f0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-3%2
#SBATCH --output=logs/dsml_phase0_b1_b4_fold0_%A_%a.log
#SBATCH --error=logs/dsml_phase0_b1_b4_fold0_%A_%a.log

set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Array index -> Phase-0 DSML ablation variant (see reports/A_STAR_2027_NOVELTY_SCAN_BROADENED.md section 4.3)
variants=(
  "mask2former_swin_small_b1_recall_reweight_fold0"
  "mask2former_swin_small_b2_sink_margin_uniform_fold0"
  "mask2former_swin_small_b3_sink_margin_symmetric_fold0"
  "mask2former_swin_small_b4_sink_margin_directed_fold0"
)
experiment_id="${variants[$SLURM_ARRAY_TASK_ID]}"
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
