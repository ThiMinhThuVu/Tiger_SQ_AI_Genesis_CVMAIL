#!/usr/bin/env bash
#SBATCH --job-name=surgigraph-a3r2-t1-test
#SBATCH --partition=gpu
# Reserve four scheduler-visible devices because external VLLM workers are not
# represented in Slurm GRES usage; select the freest device within this grant.
#SBATCH --gres=gpu:a5000:4
#SBATCH --exclude=vishc-server-1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-2%1
#SBATCH --output=logs/surgigraph_q_a3_r2_task1_test_%A_%a.log
#SBATCH --error=logs/surgigraph_q_a3_r2_task1_test_%A_%a.log

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
if [[ -z "${selected_device}" || "${selected_free}" -lt 16000 ]]; then
  echo "No allocated GPU has at least 16 GiB free: ${allocated_devices}" >&2
  exit 3
fi
export CUDA_VISIBLE_DEVICES="${selected_device}"
echo "Selected physical GPU ${selected_device} with ${selected_free} MiB free from allocation ${allocated_devices}"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    config=configs/mask2former_swin_small_surgigraph_q_a3_r2_c0_task1.yaml
    experiment=mask2former_swin_small_surgigraph_q_a3_r2_c0_task1
    ;;
  1)
    config=configs/mask2former_swin_small_surgigraph_q_a3_r2_g_task1.yaml
    experiment=mask2former_swin_small_surgigraph_q_a3_r2_g_task1
    ;;
  2)
    config=configs/mask2former_swin_small_surgigraph_q_a3_r2_gf_task1.yaml
    experiment=mask2former_swin_small_surgigraph_q_a3_r2_gf_task1
    ;;
  *)
    echo "Unsupported array index ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

model_dir="artifacts/mask2former_research/${experiment}"
for fold in 0 1 2 3 4; do
  checkpoint="${model_dir}/fold_${fold}/best.pt"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing required frozen checkpoint: ${checkpoint}" >&2
    exit 4
  fi
done

output="${model_dir}/test_task1_quantitative_r2.json"
if [[ -e "${output}" ]]; then
  echo "Refusing to overwrite existing ${output}" >&2
  exit 2
fi

python scripts/infer_mask2former_improved_test.py \
  --config "${config}" \
  --model-dir "${model_dir}" \
  --output "${output}"
