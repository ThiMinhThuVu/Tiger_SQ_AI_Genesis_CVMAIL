#!/usr/bin/env bash
#SBATCH --job-name=dsml-a0-f0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/dsml_phase0_a0_matched_env_fold0_%j.log
#SBATCH --error=logs/dsml_phase0_a0_matched_env_fold0_%j.log

set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Environment-matched A0 control for the Phase-0 DSML ablation (B1-B4 ran under
# whatever torch/transformers the executing agent had, NOT the pinned .venv).
# Fresh output dir -- never touches the real A0 artifacts under
# artifacts/mask2former_research/.
output="artifacts/dsml_phase0/mask2former_swin_small_c4_a0_nocontext_bs4_task1/fold_0"
if [[ -e "${output}/best.pt" || -e "${output}/completion.json" ]]; then
  echo "Refusing to overwrite existing result ${output}" >&2
  exit 2
fi

python scripts/train_mask2former_improved.py \
  --config configs/mask2former_swin_small_c4_a0_nocontext_bs4_task1.yaml \
  --fold 0 \
  --output-root artifacts/dsml_phase0
