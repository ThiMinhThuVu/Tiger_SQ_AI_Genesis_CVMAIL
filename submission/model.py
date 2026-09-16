"""Offline inference model for the TIGER SQ-AI 2026 submission.

The training checkpoints contain every learned parameter.  The architecture is
constructed locally so Hugging Face is never contacted at container runtime.
"""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import (
    Mask2FormerConfig,
    Mask2FormerForUniversalSegmentation,
    SwinConfig,
)


MODEL_HEIGHT = 512
MODEL_WIDTH = 896
N_FOLDS = 5
STATIONS = (
    "6L", "6R", "7L", "7R", "8", "9", "10L", "10R",
    "11L", "11R", "12L", "12R", "13L", "13R",
)

FINE_TO_COARSE = torch.tensor(
    [0, 12, 12, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 7,
     8, 9, 10, 11, 13, 5, 5, 14, 15, 15, 2, 6, 6, 2, 15],
    dtype=torch.long,
)

# Task 1 is merged/coarse and Task 2 is full/fine in the current challenge
# contract (Synapse Docker Instructions updated 2026-08-11).
TASK1_PALETTE = np.asarray(
    [
        [0, 0, 0], [0, 255, 220], [253, 189, 0], [15, 126, 87],
        [153, 76, 13], [255, 142, 142], [255, 239, 179], [255, 0, 7],
        [0, 6, 255], [50, 183, 250], [17, 139, 57], [255, 0, 204],
        [184, 61, 245], [250, 250, 55], [255, 161, 136], [173, 0, 0],
    ],
    dtype=np.uint8,
)

TASK2_PALETTE = np.asarray(
    [
        [0, 0, 0], [184, 61, 245], [134, 0, 251], [0, 255, 220],
        [131, 224, 112], [191, 224, 112], [253, 189, 0], [255, 224, 32],
        [207, 164, 117], [234, 204, 159], [15, 126, 87], [153, 76, 13],
        [51, 221, 255], [255, 142, 142], [255, 239, 179], [255, 0, 7],
        [0, 6, 255], [50, 183, 250], [17, 139, 57], [255, 0, 204],
        [250, 250, 55], [255, 136, 178], [255, 73, 162], [255, 161, 136],
        [173, 0, 0], [128, 128, 128], [245, 147, 49], [218, 161, 69],
        [255, 217, 129], [219, 219, 56], [201, 138, 118],
    ],
    dtype=np.uint8,
)


def build_mask2former_config() -> Mask2FormerConfig:
    """Recreate facebook/mask2former-swin-small-coco-panoptic offline."""
    backbone = SwinConfig(
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        drop_path_rate=0.3,
        out_features=["stage1", "stage2", "stage3", "stage4"],
    )
    labels = {index: str(index) for index in range(31)}
    return Mask2FormerConfig(
        num_labels=31,
        id2label=labels,
        label2id={value: key for key, value in labels.items()},
        backbone_config=backbone,
        feature_size=256,
        mask_feature_size=256,
        hidden_dim=256,
        encoder_feedforward_dim=1024,
        encoder_layers=6,
        decoder_layers=10,
        dim_feedforward=2048,
        num_attention_heads=8,
        num_queries=100,
        common_stride=4,
        feature_strides=[4, 8, 16, 32],
        use_auxiliary_loss=True,
    )


class InferenceMask2Former(nn.Module):
    """Inference-only equivalent of the trained ImprovedMask2Former."""

    def __init__(self) -> None:
        super().__init__()
        self.segmenter = Mask2FormerForUniversalSegmentation(build_mask2former_config())
        self.coarse_head = nn.Conv2d(256, 16, kernel_size=1)
        self.presence_head = nn.Linear(256, 31)
        self.register_buffer("fine_to_coarse", FINE_TO_COARSE.clone())

    def forward(self, pixel_values: Tensor) -> dict[str, Tensor]:
        output = self.segmenter(
            pixel_values=pixel_values,
            output_auxiliary_logits=False,
        )
        feature = output.pixel_decoder_last_hidden_state
        class_probability = output.class_queries_logits.softmax(-1)[..., :-1]
        mask_probability = output.masks_queries_logits.sigmoid()
        fine = torch.einsum("bqc,bqhw->bchw", class_probability, mask_probability)
        fine = fine / fine.sum(1, keepdim=True).clamp_min(1e-6)
        return {
            "fine_probability": fine,
            "coarse_logits": self.coarse_head(feature),
            "presence_logits": self.presence_head(feature.mean((-2, -1))),
            "feature": feature,
        }


class VisibilityHead(nn.Module):
    def __init__(self, channels: int = 256) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, len(STATIONS)),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.layers(features)


def _load_checkpoint(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required checkpoint is missing: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Invalid checkpoint payload: {path}")
    return checkpoint


def load_segmentation_model(path: Path, device: torch.device) -> InferenceMask2Former:
    checkpoint = _load_checkpoint(path)
    state = checkpoint.get("model")
    if not isinstance(state, dict):
        raise KeyError(f"Checkpoint has no model state: {path}")

    # Training-only clinical loss buffers are intentionally excluded.  All
    # inference parameters must still match exactly.
    state = {
        key: value
        for key, value in state.items()
        if not key.startswith("segmenter.criterion.")
    }
    model = InferenceMask2Former()
    incompatible = model.load_state_dict(state, strict=False)
    expected_missing = {"segmenter.criterion.empty_weight"}
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint/model mismatch for {path}: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    # The loss module is never used in inference and need not remain resident.
    model.segmenter.criterion = None
    return model.to(device).eval()


def load_visibility_head(path: Path, device: torch.device) -> VisibilityHead:
    checkpoint = _load_checkpoint(path)
    channels = int(checkpoint.get("feature_channels", 0))
    if channels != 256:
        raise ValueError(f"Expected 256 visibility feature channels, got {channels}: {path}")
    head = VisibilityHead(channels)
    head.load_state_dict(checkpoint["visibility_head"], strict=True)
    return head.to(device).eval()


def load_ensemble(checkpoint_root: Path, device: torch.device):
    models: list[InferenceMask2Former] = []
    heads: list[VisibilityHead] = []
    for fold in range(N_FOLDS):
        print(f"loading fold {fold + 1}/{N_FOLDS}", flush=True)
        models.append(
            load_segmentation_model(
                checkpoint_root / "segmentation" / f"fold_{fold}" / "best.pt",
                device,
            )
        )
        heads.append(
            load_visibility_head(
                checkpoint_root / "visibility" / f"fold_{fold}" / "best_visibility.pt",
                device,
            )
        )
    return models, heads


def preprocess(image) -> Tensor:
    resized = image.resize((MODEL_WIDTH, MODEL_HEIGHT), resample=2)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = (array - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)) / np.asarray(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    return torch.from_numpy(array.transpose(2, 0, 1).copy()).unsqueeze(0)


def _presence_modulation(probability: Tensor, logits: Tensor) -> Tensor:
    scale = 0.25 + 0.75 * logits.sigmoid()
    output = probability * scale[:, :, None, None]
    return output / output.sum(1, keepdim=True).clamp_min(1e-6)


def _fine_to_coarse(probability: Tensor, mapping: Tensor) -> Tensor:
    matrix = F.one_hot(mapping.long(), num_classes=16).T.to(probability)
    return torch.einsum("cf,bfhw->bchw", matrix, probability)


@torch.inference_mode()
def predict_ensemble(
    image,
    models: list[InferenceMask2Former],
    heads: list[VisibilityHead],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(models) != N_FOLDS or len(heads) != N_FOLDS:
        raise ValueError(f"Expected {N_FOLDS} model/head pairs")
    tensor = preprocess(image).to(device, non_blocking=device.type == "cuda")
    fine_sum = coarse_sum = station_sum = None
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else nullcontext()
    )

    for model, head in zip(models, heads):
        with autocast:
            result = model(tensor)
            fine = _presence_modulation(
                result["fine_probability"], result["presence_logits"]
            )
            derived = _fine_to_coarse(fine, model.fine_to_coarse)
            direct = result["coarse_logits"].softmax(1)
            coarse = 0.5 * direct + 0.5 * derived
            stations = head(result["feature"].mean((-2, -1))).sigmoid()

        fine = fine.float()
        coarse = coarse.float()
        stations = stations.float()
        fine_sum = fine if fine_sum is None else fine_sum + fine
        coarse_sum = coarse if coarse_sum is None else coarse_sum + coarse
        station_sum = stations if station_sum is None else station_sum + stations

    fine_probability = fine_sum / N_FOLDS
    coarse_probability = coarse_sum / N_FOLDS
    output_size = (image.height, image.width)
    fine_ids = F.interpolate(
        fine_probability, output_size, mode="bilinear", align_corners=False
    ).argmax(1)[0]
    coarse_ids = F.interpolate(
        coarse_probability, output_size, mode="bilinear", align_corners=False
    ).argmax(1)[0]
    station_probability = (station_sum / N_FOLDS)[0]
    return (
        coarse_ids.cpu().numpy().astype(np.uint8),
        fine_ids.cpu().numpy().astype(np.uint8),
        station_probability.cpu().numpy().clip(0.0, 1.0),
    )
