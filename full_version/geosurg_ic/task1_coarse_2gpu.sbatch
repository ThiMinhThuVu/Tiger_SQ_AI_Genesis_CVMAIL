#!/usr/bin/env bash
#SBATCH --job-name=tiger_t1_geosurg
#SBATCH --partition=gpu
#SBATCH --nodelist=vishc-server-2
#SBATCH --gres=gpu:a5000:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=full_version/geosurg_ic/artifacts/task1_coarse_v1/controller_%j.log
#SBATCH --error=full_version/geosurg_ic/artifacts/task1_coarse_v1/controller_%j.log
set -euo pipefail
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
out=full_version/geosurg_ic/artifacts/task1_coarse_v1
# GRES does not reliably hide devices on this cluster: pin physical GPUs 5 and 7.
gpus=(GPU-fb071ee2-e6bf-42c9-b49f-296e5524513f GPU-05f5881b-fd24-0564-185f-9f95712d1997)
# IC weight 0.05: matches Task-2 relative IC/supervised gradient scale
# (Task-1 audit median ratio 1.78 at weight 1; Task-2 used 0.02 at 4.99).
run() {
    local gpu="$1" arm="$2" fold="$3"
    echo "Starting arm=${arm} fold=${fold} gpu=${gpu} date=$(date -u +%FT%TZ)"
    CUDA_VISIBLE_DEVICES="${gpu}" timeout 3600 .venv/bin/python -u full_version/geosurg_ic/run_task1_coarse.py \
        --arm "${arm}" --fold "${fold}" --steps 192 --ic-every 2 --ic-weight 0.05 --output "${out}" \
        >"${out}/${arm}_fold${fold}.log" 2>&1
    echo "Completed arm=${arm} fold=${fold} date=$(date -u +%FT%TZ)"
}
worker_geo() { for f in 0 1 2 3 4; do run "${gpus[0]}" geo "$f"; done; }
worker_aug() {
    for f in 0 1 2 3 4; do run "${gpus[1]}" aug "$f"; done
    for f in 0 1 2 3 4; do run "${gpus[1]}" baseline "$f"; done
}
worker_geo & worker_aug &
wait
.venv/bin/python full_version/geosurg_ic/summarize_task1_coarse.py --root "${out}"
