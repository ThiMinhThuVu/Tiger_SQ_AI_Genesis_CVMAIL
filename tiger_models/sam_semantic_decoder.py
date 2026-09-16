from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def _conv_block(channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        nn.GroupNorm(16, channels),
        nn.GELU(),
        nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        nn.GroupNorm(16, channels),
        nn.GELU(),
    )


class SAMSemanticDecoder(nn.Module):
    def __init__(
        self,
        input_channels: Sequence[int] = (256, 256, 256, 256),
        decoder_channels: int = 128,
        fine_classes: int = 31,
        coarse_classes: int | None = None,
    ) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            nn.Conv2d(channels, decoder_channels, 1) for channels in input_channels
        )
        self.refine = nn.ModuleList(_conv_block(decoder_channels) for _ in input_channels)
        self.fine_head = nn.Conv2d(decoder_channels, fine_classes, 1)
        self.coarse_head = (
            nn.Conv2d(decoder_channels, coarse_classes, 1)
            if coarse_classes is not None else None
        )

    def forward(
        self,
        features: Sequence[torch.Tensor],
        output_size: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        if len(features) != len(self.projections):
            raise ValueError(f"Expected {len(self.projections)} features, got {len(features)}")
        x = self.refine[-1](self.projections[-1](features[-1]))
        for index in range(len(features) - 2, -1, -1):
            x = F.interpolate(x, size=features[index].shape[-2:], mode="bilinear", align_corners=False)
            x = self.refine[index](x + self.projections[index](features[index]))
        fine = F.interpolate(self.fine_head(x), size=output_size, mode="bilinear", align_corners=False)
        output = {"fine_logits": fine}
        if self.coarse_head is not None:
            output["coarse_logits"] = F.interpolate(
                self.coarse_head(x), size=output_size, mode="bilinear", align_corners=False
            )
        return output
