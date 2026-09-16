FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME=/opt/huggingface \
    TORCH_HOME=/opt/checkpoints TOKENIZERS_PARALLELISM=false \
    PYTHONPATH=/opt/algorithm/vendor
WORKDIR /opt/algorithm

COPY model.py /opt/algorithm/model_base.py
COPY contract.py validate_output.py /opt/algorithm/
COPY ver1/model.py ver1/predict.py /opt/algorithm/
COPY --from=vendor / /opt/algorithm/vendor/

COPY --from=task1 fold_0/best.pt /opt/checkpoints/task1/fold_0/best.pt
COPY --from=task1 fold_1/best.pt /opt/checkpoints/task1/fold_1/best.pt
COPY --from=task1 fold_2/best.pt /opt/checkpoints/task1/fold_2/best.pt
COPY --from=task1 fold_3/best.pt /opt/checkpoints/task1/fold_3/best.pt
COPY --from=task1 fold_4/best.pt /opt/checkpoints/task1/fold_4/best.pt
COPY --from=task2 fold_0/best.pt /opt/checkpoints/task2/fold_0/best.pt
COPY --from=task2 fold_1/best.pt /opt/checkpoints/task2/fold_1/best.pt
COPY --from=task2 fold_2/best.pt /opt/checkpoints/task2/fold_2/best.pt
COPY --from=task2 fold_3/best.pt /opt/checkpoints/task2/fold_3/best.pt
COPY --from=task2 fold_4/best.pt /opt/checkpoints/task2/fold_4/best.pt
COPY --from=task3 fold_0/best_visibility.pt /opt/checkpoints/task3/fold_0/best_visibility.pt
COPY --from=task3 fold_1/best_visibility.pt /opt/checkpoints/task3/fold_1/best_visibility.pt
COPY --from=task3 fold_2/best_visibility.pt /opt/checkpoints/task3/fold_2/best_visibility.pt
COPY --from=task3 fold_3/best_visibility.pt /opt/checkpoints/task3/fold_3/best_visibility.pt
COPY --from=task3 fold_4/best_visibility.pt /opt/checkpoints/task3/fold_4/best_visibility.pt

ENTRYPOINT ["python", "/opt/algorithm/predict.py"]

