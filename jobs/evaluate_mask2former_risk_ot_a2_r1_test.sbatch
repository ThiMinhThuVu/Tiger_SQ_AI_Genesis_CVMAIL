#!/usr/bin/env bash
#SBATCH --job-name=m2f-risk-ot-a2-r1-test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/mask2former_risk_ot_a2_r1_test_%j.log
#SBATCH --error=logs/mask2former_risk_ot_a2_r1_test_%j.log

set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate

output=artifacts/mask2former_research/mask2former_swin_small_risk_ot_a2_r1/test_quantitative.json
if [[ -e "${output}" ]]; then
  echo "Refusing to overwrite existing ${output}" >&2
  exit 2
fi

python scripts/infer_mask2former_improved_test.py \
  --config configs/mask2former_swin_small_risk_ot_a2_r1.yaml \
  --model-dir artifacts/mask2former_research/mask2former_swin_small_risk_ot_a2_r1 \
  --output "${output}"
