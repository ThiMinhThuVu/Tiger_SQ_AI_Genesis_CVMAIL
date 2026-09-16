#!/usr/bin/env bash
#SBATCH --job-name=c5-rbcot-cv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a5000:4
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --array=1-4%1
#SBATCH --output=logs/mask2former_c5_rb_cot_cv_%A_%a.log
#SBATCH --error=logs/mask2former_c5_rb_cot_cv_%A_%a.log

set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

allocated_devices="${CUDA_VISIBLE_DEVICES}"
IFS=',' read -r -a device_ids <<< "${allocated_devices}"
selected_device=""
selected_free=-1
for device_id in "${device_ids[@]}"; do
  free_memory="$(nvidia-smi -i "${device_id}" --query-gpu=memory.free --format=csv,noheader,nounits | head -1)"
  if (( free_memory > selected_free )); then
    selected_device="${device_id}"
    selected_free="${free_memory}"
  fi
done
if [[ -z "${selected_device}" || "${selected_free}" -lt 22000 ]]; then
  echo "No allocated GPU has at least 22 GiB free: ${allocated_devices}" >&2
  exit 3
fi
export CUDA_VISIBLE_DEVICES="${selected_device}"

fold="${SLURM_ARRAY_TASK_ID}"
output="artifacts/mask2former_research/mask2former_swin_small_c5_rb_cot_bs4_task1/fold_${fold}"
if [[ -e "${output}/best.pt" || -e "${output}/completion.json" ]]; then
  echo "Refusing to overwrite completed result ${output}" >&2
  exit 2
fi

python scripts/train_mask2former_improved.py \
  --config configs/mask2former_swin_small_c5_rb_cot_bs4_task1.yaml \
  --fold "${fold}" \
  --output-root artifacts/mask2former_research
