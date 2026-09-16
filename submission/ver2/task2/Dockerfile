FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME=/opt/huggingface \
    TORCH_HOME=/opt/checkpoints TOKENIZERS_PARALLELISM=false \
    PYTHONPATH=/opt/algorithm/vendor
WORKDIR /opt/algorithm

COPY model.py /opt/algorithm/model_base.py
COPY ver1/model.py /opt/algorithm/model_ver1.py
COPY ver2/model.py ver2/task2/predict.py ver2/task2/validate.py /opt/algorithm/
COPY --from=vendor / /opt/algorithm/vendor/

COPY --from=task2 fold_0/best.pt /opt/checkpoints/task2/fold_0/best.pt
COPY --from=task2 fold_1/best.pt /opt/checkpoints/task2/fold_1/best.pt
COPY --from=task2 fold_2/best.pt /opt/checkpoints/task2/fold_2/best.pt
COPY --from=task2 fold_3/best.pt /opt/checkpoints/task2/fold_3/best.pt
COPY --from=task2 fold_4/best.pt /opt/checkpoints/task2/fold_4/best.pt

ENTRYPOINT ["python", "/opt/algorithm/predict.py"]
