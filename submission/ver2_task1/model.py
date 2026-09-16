"""Standalone Task 1 inference helpers for the corrected D517 checkpoints."""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

try:
    from model_base import MODEL_HEIGHT, MODEL_WIDTH, N_FOLDS, TASK1_PALETTE
    from model_ver1 import load_task1
except ModuleNotFoundError:  # Repository-side verification.
    from submission.model import MODEL_HEIGHT, MODEL_WIDTH, N_FOLDS, TASK1_PALETTE
    from submission.ver1.model import load_task1


def load_task1_ensemble(checkpoint_root: Path, device: torch.device):
    models = []
    for fold in range(N_FOLDS):
        print(f"loading Task 1 D517 fold {fold + 1}/{N_FOLDS}", flush=True)
        models.append(load_task1(checkpoint_root / f"fold_{fold}/best.pt", device))
    return models


def _preprocess(image) -> torch.Tensor:
    resized = image.resize((MODEL_WIDTH, MODEL_HEIGHT), resample=2)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = (array - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)) / np.asarray(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    return torch.from_numpy(array.transpose(2, 0, 1).copy()).unsqueeze(0)


@torch.inference_mode()
def predict_task1(image, models, device: torch.device) -> np.ndarray:
    if len(models) != N_FOLDS:
        raise ValueError(f"Expected {N_FOLDS} Task 1 models, got {len(models)}")
    tensor = _preprocess(image).to(device, non_blocking=device.type == "cuda")
    total = None
    autocast = torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
    for model in models:
        with autocast:
            probability = model(tensor)
        probability = probability.float()
        total = probability if total is None else total + probability
    output_size = (image.height, image.width)
    ids = F.interpolate(total / N_FOLDS, output_size, mode="bilinear", align_corners=False).argmax(1)[0]
    return ids.cpu().numpy().astype(np.uint8)
