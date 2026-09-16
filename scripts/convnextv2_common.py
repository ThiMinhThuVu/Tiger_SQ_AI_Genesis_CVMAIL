#!/usr/bin/env python3
"""ConvNeXt-V2 multitask model used by the TIGER baseline runner."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNeXtV2Multitask(nn.Module):
    """ConvNeXt-V2 encoder with fine segmentation and visibility heads.

    Task 2 is intentionally derived from Task 1 using the repository's
    verified fine-to-coarse mapping, matching the SAM baseline convention.
    """

    def __init__(self, model_name: str, fine_classes: int = 31,
                 visibility_classes: int = 14, pretrained: bool = True,
                 decoder_channels: int = 128) -> None:
        super().__init__()
        import timm

        self.encoder = timm.create_model(
            model_name, pretrained=pretrained, features_only=True,
        )
        channels = list(self.encoder.feature_info.channels())
        self.lateral = nn.ModuleList([nn.Conv2d(c, decoder_channels, 1) for c in channels])
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1),
            nn.BatchNorm2d(decoder_channels), nn.GELU(),
            nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1),
            nn.BatchNorm2d(decoder_channels), nn.GELU(),
        )
        self.fine_head = nn.Conv2d(decoder_channels, fine_classes, 1)
        self.visibility_head = nn.Sequential(
            nn.LayerNorm(channels[-1]), nn.Linear(channels[-1], 256), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(256, visibility_classes),
        )

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(image)
        x = self.lateral[-1](features[-1])
        for index in range(len(features) - 2, -1, -1):
            x = F.interpolate(x, size=features[index].shape[-2:], mode="bilinear", align_corners=False)
            x = x + self.lateral[index](features[index])
        fine = self.fine_head(self.fuse(x))
        fine = F.interpolate(fine, size=image.shape[-2:], mode="bilinear", align_corners=False)
        pooled = features[-1].mean(dim=(-2, -1))
        return {"fine_logits": fine, "visibility_logits": self.visibility_head(pooled)}


def multitask_loss(output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor],
                   visibility_pos_weight: torch.Tensor) -> dict[str, torch.Tensor]:
    fine = F.cross_entropy(output["fine_logits"], batch["fine"].long())
    probability = output["fine_logits"].softmax(1)
    one_hot = F.one_hot(batch["fine"].long(), output["fine_logits"].shape[1]).permute(0, 3, 1, 2).float()
    intersection = (probability * one_hot).sum((0, 2, 3))
    denominator = probability.sum((0, 2, 3)) + one_hot.sum((0, 2, 3))
    dice = 1.0 - ((2 * intersection + 1e-5) / (denominator + 1e-5)).mean()
    visibility = F.binary_cross_entropy_with_logits(
        output["visibility_logits"], batch["visibility"].float(),
        pos_weight=visibility_pos_weight,
    )
    return {"fine": fine, "dice": dice, "visibility": visibility,
            "total": fine + dice + visibility}
