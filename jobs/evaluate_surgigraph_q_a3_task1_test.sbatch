#!/usr/bin/env bash
#SBATCH --job-name=surgigraph-a3-t1-test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/surgigraph_q_a3_task1_test_%A_%a.log
#SBATCH --error=logs/surgigraph_q_a3_task1_test_%A_%a.log

set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    config=configs/mask2former_swin_small_surgigraph_q_a3_c0_task1.yaml
    experiment=mask2former_swin_small_surgigraph_q_a3_c0_task1
    ;;
  1)
    config=configs/mask2former_swin_small_surgigraph_q_a3_m_task1.yaml
    experiment=mask2former_swin_small_surgigraph_q_a3_m_task1
    ;;
  2)
    config=configs/mask2former_swin_small_surgigraph_q_a3_mf_task1.yaml
    experiment=mask2former_swin_small_surgigraph_q_a3_mf_task1
    ;;
  *)
    echo "Unsupported array index ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

model_dir="artifacts/mask2former_research/${experiment}"
output="${model_dir}/test_task1_quantitative.json"
if [[ -e "${output}" ]]; then
  echo "Refusing to overwrite existing ${output}" >&2
  exit 2
fi

python scripts/infer_mask2former_improved_test.py \
  --config "${config}" \
  --model-dir "${model_dir}" \
  --output "${output}"
