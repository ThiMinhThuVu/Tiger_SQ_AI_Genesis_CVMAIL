#!/usr/bin/env bash
#SBATCH --job-name=surgigraph-q-a3-t1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=60:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/surgigraph_q_a3_task1_%A_%a.log
#SBATCH --error=logs/surgigraph_q_a3_task1_%A_%a.log

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

output="artifacts/mask2former_research/${experiment}"
if [[ -e "${output}" ]]; then
  echo "Refusing to overwrite existing ${output}" >&2
  exit 2
fi

python scripts/train_mask2former_improved.py \
  --config "${config}" \
  --run-all-folds \
  --output-root artifacts/mask2former_research
