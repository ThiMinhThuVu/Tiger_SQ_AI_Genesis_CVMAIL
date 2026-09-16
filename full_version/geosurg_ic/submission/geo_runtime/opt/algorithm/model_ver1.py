"""Ver1 inference: independent Task-1 coarse and Task-2 fine ensembles."""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import Mask2FormerForUniversalSegmentation

try:
    from model_base import (
        MODEL_HEIGHT, MODEL_WIDTH, N_FOLDS, STATIONS, TASK1_PALETTE, TASK2_PALETTE,
        InferenceMask2Former, VisibilityHead, _load_checkpoint,
        _presence_modulation, build_mask2former_config,
    )
except ModuleNotFoundError:  # Repository-side verification before Docker COPY.
    from submission.model import (
        MODEL_HEIGHT, MODEL_WIDTH, N_FOLDS, STATIONS, TASK1_PALETTE, TASK2_PALETTE,
        InferenceMask2Former, VisibilityHead, _load_checkpoint,
        _presence_modulation, build_mask2former_config,
    )


class CoarseMask2Former(Mask2FormerForUniversalSegmentation):
    """Exact inference structure of the independently trained 16-class model."""

    def __init__(self) -> None:
        config = build_mask2former_config()
        config.num_labels = 16
        config.id2label = {index: str(index) for index in range(16)}
        config.label2id = {value: key for key, value in config.id2label.items()}
        super().__init__(config)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        output = super().forward(pixel_values=pixel_values, output_auxiliary_logits=False)
        classes = output.class_queries_logits.softmax(-1)[..., :-1]
        masks = output.masks_queries_logits.sigmoid()
        return torch.einsum("bqc,bqhw->bchw", classes, masks)


def _load_model_state(model: nn.Module, path: Path, prefix: str = "") -> nn.Module:
    checkpoint = _load_checkpoint(path)
    state = checkpoint.get("model")
    if not isinstance(state, dict):
        raise KeyError(f"Checkpoint has no model state: {path}")
    state = {key: value for key, value in state.items()
             if "criterion." not in key}
    incompatible = model.load_state_dict(state, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing = [key for key in incompatible.missing_keys if "criterion." not in key]
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint/model mismatch for {path}: missing={missing}, unexpected={unexpected}")
    target = getattr(model, prefix) if prefix else model
    if hasattr(target, "criterion"):
        target.criterion = None
    return model


def load_task1(path: Path, device: torch.device) -> CoarseMask2Former:
    model = _load_model_state(CoarseMask2Former(), path)
    return model.to(device).eval()


def load_task2(path: Path, device: torch.device) -> InferenceMask2Former:
    model = _load_model_state(InferenceMask2Former(), path, "segmenter")
    return model.to(device).eval()


def load_visibility(path: Path, device: torch.device) -> VisibilityHead:
    checkpoint = _load_checkpoint(path)
    state = checkpoint.get("visibility_head")
    if not isinstance(state, dict):
        raise KeyError(f"Checkpoint has no visibility_head: {path}")
    # Ver1 training checkpoints omit the redundant feature_channels field.
    channels = int(checkpoint.get("feature_channels", state["layers.0.weight"].numel()))
    head = VisibilityHead(channels)
    head.load_state_dict(state, strict=True)
    return head.to(device).eval()


def load_ensemble(checkpoint_root: Path, device: torch.device):
    task1_models, task2_models, heads = [], [], []
    for fold in range(N_FOLDS):
        print(f"loading ver1 fold {fold + 1}/{N_FOLDS}", flush=True)
        task1_models.append(load_task1(checkpoint_root / "task1" / f"fold_{fold}/best.pt", device))
        task2_models.append(load_task2(checkpoint_root / "task2" / f"fold_{fold}/best.pt", device))
        heads.append(load_visibility(checkpoint_root / "task3" / f"fold_{fold}/best_visibility.pt", device))
    return task1_models, task2_models, heads


def preprocess(image) -> torch.Tensor:
    resized = image.resize((MODEL_WIDTH, MODEL_HEIGHT), resample=2)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = (array - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)) / np.asarray(
        [0.229, 0.224, 0.225], dtype=np.float32)
    return torch.from_numpy(array.transpose(2, 0, 1).copy()).unsqueeze(0)


@torch.inference_mode()
def predict_ensemble(image, task1_models, task2_models, heads, device):
    if not (len(task1_models) == len(task2_models) == len(heads) == N_FOLDS):
        raise ValueError(f"Expected {N_FOLDS} models for every task")
    tensor = preprocess(image).to(device, non_blocking=device.type == "cuda")
    task1_sum = task2_sum = station_sum = None
    autocast = torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
    for task1_model, task2_model, head in zip(task1_models, task2_models, heads):
        with autocast:
            coarse = task1_model(tensor)
            result = task2_model(tensor)
            fine = _presence_modulation(result["fine_probability"], result["presence_logits"])
            stations = head(result["feature"].mean((-2, -1))).sigmoid()
        coarse, fine, stations = coarse.float(), fine.float(), stations.float()
        task1_sum = coarse if task1_sum is None else task1_sum + coarse
        task2_sum = fine if task2_sum is None else task2_sum + fine
        station_sum = stations if station_sum is None else station_sum + stations
    output_size = (image.height, image.width)
    task1_ids = F.interpolate(task1_sum / N_FOLDS, output_size, mode="bilinear", align_corners=False).argmax(1)[0]
    task2_ids = F.interpolate(task2_sum / N_FOLDS, output_size, mode="bilinear", align_corners=False).argmax(1)[0]
    station_probability = (station_sum / N_FOLDS)[0]
    return (task1_ids.cpu().numpy().astype(np.uint8), task2_ids.cpu().numpy().astype(np.uint8),
            station_probability.cpu().numpy().clip(0.0, 1.0))
