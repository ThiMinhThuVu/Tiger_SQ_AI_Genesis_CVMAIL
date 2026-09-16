"""Task-specific ver2 inference helpers."""
from __future__ import annotations

from contextlib import nullcontext
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from model_ver1 import (
    N_FOLDS,
    STATIONS,
    TASK2_PALETTE,
    _presence_modulation,
    load_task2,
    load_visibility,
)

TASK2_HEIGHT = int(os.environ.get("TIGERSQAI_TASK2_HEIGHT", "640"))
TASK2_WIDTH = int(os.environ.get("TIGERSQAI_TASK2_WIDTH", "1120"))
TASK3_HEIGHT = int(os.environ.get("TIGERSQAI_TASK3_HEIGHT", "512"))
TASK3_WIDTH = int(os.environ.get("TIGERSQAI_TASK3_WIDTH", "896"))


def preprocess(image, height: int, width: int) -> torch.Tensor:
    resized = image.resize((width, height), resample=2)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = (array - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)) / np.asarray(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    return torch.from_numpy(array.transpose(2, 0, 1).copy()).unsqueeze(0)


def load_task2_ensemble(checkpoint_root: Path, device: torch.device):
    models = []
    for fold in range(N_FOLDS):
        print(f"loading Task 2 P2 fold {fold + 1}/{N_FOLDS}", flush=True)
        models.append(load_task2(checkpoint_root / f"fold_{fold}/best.pt", device))
    return models


def load_task3_ensemble(
    encoder_root: Path, head_root: Path, device: torch.device
):
    models, heads = [], []
    for fold in range(N_FOLDS):
        print(f"loading Task 3 P0 fold {fold + 1}/{N_FOLDS}", flush=True)
        models.append(load_task2(encoder_root / f"fold_{fold}/best.pt", device))
        heads.append(load_visibility(head_root / f"fold_{fold}/best_visibility.pt", device))
    return models, heads


def _autocast(device: torch.device):
    return (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else nullcontext()
    )


@torch.inference_mode()
def predict_task2(image, models, device: torch.device) -> np.ndarray:
    if len(models) != N_FOLDS:
        raise ValueError(f"Expected {N_FOLDS} Task 2 models")
    tensor = preprocess(image, height=TASK2_HEIGHT, width=TASK2_WIDTH).to(
        device, non_blocking=device.type == "cuda"
    )
    fine_sum = None
    for model in models:
        with _autocast(device):
            result = model(tensor)
            fine = _presence_modulation(
                result["fine_probability"], result["presence_logits"]
            )
        fine = fine.float()
        fine_sum = fine if fine_sum is None else fine_sum + fine
    output_size = (image.height, image.width)
    ids = F.interpolate(
        fine_sum / N_FOLDS, output_size, mode="bilinear", align_corners=False
    ).argmax(1)[0]
    return ids.cpu().numpy().astype(np.uint8)


@torch.inference_mode()
def predict_task3(image, models, heads, device: torch.device) -> np.ndarray:
    if len(models) != N_FOLDS or len(heads) != N_FOLDS:
        raise ValueError(f"Expected {N_FOLDS} Task 3 model/head pairs")
    tensor = preprocess(image, height=TASK3_HEIGHT, width=TASK3_WIDTH).to(
        device, non_blocking=device.type == "cuda"
    )
    station_sum = None
    for model, head in zip(models, heads):
        with _autocast(device):
            result = model(tensor)
            stations = head(result["feature"].mean((-2, -1))).sigmoid()
        stations = stations.float()
        station_sum = stations if station_sum is None else station_sum + stations
    return (station_sum / N_FOLDS)[0].cpu().numpy().clip(0.0, 1.0)
