#!/usr/bin/env bash
#SBATCH --job-name=cv2base-m2f-test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/convnextv2_base_mask2former_test_%j.log
#SBATCH --error=logs/convnextv2_base_mask2former_test_%j.log
set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate

output=artifacts/new_baselines/convnextv2_base_mask2former/test_quantitative.json
if [[ -e "${output}" ]]; then
  echo "Refusing to overwrite existing ${output}" >&2
  exit 2
fi

python scripts/evaluate_extra_multitask_test.py \
  --model mask2former \
  --config configs/convnextv2_base_mask2former.yaml \
  --model-dir artifacts/new_baselines/convnextv2_base_mask2former \
  --output "${output}"
