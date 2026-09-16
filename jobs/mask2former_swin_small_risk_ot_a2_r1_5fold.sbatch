#!/usr/bin/env bash
#SBATCH --job-name=m2f-risk-ot-a2-r1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=60:00:00
#SBATCH --output=logs/mask2former_swin_small_risk_ot_a2_r1_%j.log
#SBATCH --error=logs/mask2former_swin_small_risk_ot_a2_r1_%j.log
set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate

output=artifacts/mask2former_research/mask2former_swin_small_risk_ot_a2_r1
if [[ -e "${output}" ]]; then
  echo "Refusing to overwrite existing ${output}" >&2
  exit 2
fi

python scripts/train_mask2former_improved.py \
  --config configs/mask2former_swin_small_risk_ot_a2_r1.yaml \
  --run-all-folds \
  --output-root artifacts/mask2former_research
