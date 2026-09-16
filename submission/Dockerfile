FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    HF_HOME=/opt/huggingface \
    TORCH_HOME=/opt/checkpoints \
    TOKENIZERS_PARALLELISM=false \
    PYTHONPATH=/opt/algorithm/vendor

WORKDIR /opt/algorithm

COPY model.py contract.py predict.py validate_output.py /opt/algorithm/

# build.sh resolves binary wheels for the base image's CPython 3.11 ABI into
# this named context.  COPY-only assembly also works with unprivileged
# BuildKit environments that cannot execute Dockerfile RUN instructions.
COPY --from=vendor / /opt/algorithm/vendor/

# These two named build contexts are supplied by build.sh.  Every learned
# weight is baked into the image; the base COCO weights are not needed.
COPY --from=segmentation fold_0/best.pt /opt/checkpoints/segmentation/fold_0/best.pt
COPY --from=segmentation fold_1/best.pt /opt/checkpoints/segmentation/fold_1/best.pt
COPY --from=segmentation fold_2/best.pt /opt/checkpoints/segmentation/fold_2/best.pt
COPY --from=segmentation fold_3/best.pt /opt/checkpoints/segmentation/fold_3/best.pt
COPY --from=segmentation fold_4/best.pt /opt/checkpoints/segmentation/fold_4/best.pt
COPY --from=visibility fold_0/best_visibility.pt /opt/checkpoints/visibility/fold_0/best_visibility.pt
COPY --from=visibility fold_1/best_visibility.pt /opt/checkpoints/visibility/fold_1/best_visibility.pt
COPY --from=visibility fold_2/best_visibility.pt /opt/checkpoints/visibility/fold_2/best_visibility.pt
COPY --from=visibility fold_3/best_visibility.pt /opt/checkpoints/visibility/fold_3/best_visibility.pt
COPY --from=visibility fold_4/best_visibility.pt /opt/checkpoints/visibility/fold_4/best_visibility.pt

ENTRYPOINT ["python", "/opt/algorithm/predict.py"]
