#!/usr/bin/env bash
#SBATCH --job-name=c4-ablate-test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-3%1
#SBATCH --output=logs/mask2former_c4_ablation_test_%A_%a.log
#SBATCH --error=logs/mask2former_c4_ablation_test_%A_%a.log

set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate

configs=(
  configs/mask2former_swin_small_c4_a0_nocontext_bs4_task1.yaml
  configs/mask2former_swin_small_c4_a1_context_head_bs4_task1.yaml
  configs/mask2former_swin_small_c4_a2_context_mass_bs4_task1.yaml
  configs/mask2former_swin_small_c4_a3_context_fp_bs4_task1.yaml
)
experiments=(
  mask2former_swin_small_c4_a0_nocontext_bs4_task1
  mask2former_swin_small_c4_a1_context_head_bs4_task1
  mask2former_swin_small_c4_a2_context_mass_bs4_task1
  mask2former_swin_small_c4_a3_context_fp_bs4_task1
)

index="${SLURM_ARRAY_TASK_ID}"
config="${configs[$index]}"
model_dir="artifacts/mask2former_research/${experiments[$index]}"
output="${model_dir}/test_task1_quantitative.json"

for fold in {0..4}; do
  for required in best.pt completion.json; do
    path="${model_dir}/fold_${fold}/${required}"
    if [[ ! -f "${path}" ]]; then
      echo "Incomplete five-fold result: missing ${path}" >&2
      exit 3
    fi
  done
done

if [[ -e "${output}" ]]; then
  echo "Refusing to overwrite existing ${output}" >&2
  exit 2
fi

echo "Evaluating ${experiments[$index]} on the held-out five-fold test split"
python scripts/infer_mask2former_improved_test.py \
  --config "${config}" \
  --model-dir "${model_dir}" \
  --output "${output}"
