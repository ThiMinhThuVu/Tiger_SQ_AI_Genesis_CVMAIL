from __future__ import annotations

import torch
from torch import nn

from .sam2_encoder import SAM2HieraEncoder
from .sam_semantic_decoder import SAMSemanticDecoder


class MultitaskSAM(nn.Module):
    def __init__(
        self,
        checkpoint_path: str | None = None,
        finetune: str = "partial",
        gradient_checkpointing: bool = True,
        decoder_channels: int = 128,
        coarse_head: bool = False,
        encoder: nn.Module | None = None,
        config_name: str = "configs/sam2.1/sam2.1_hiera_t.yaml",
    ) -> None:
        super().__init__()
        if encoder is None:
            if checkpoint_path is None:
                raise ValueError("checkpoint_path is required for the real SAM 2 encoder")
            encoder = SAM2HieraEncoder(
                checkpoint_path,
                finetune=finetune,
                gradient_checkpointing=gradient_checkpointing,
                config_name=config_name,
            )
        self.encoder = encoder
        input_channels = tuple(getattr(encoder, "feature_channels", (256, 256, 256, 256)))
        self.decoder = SAMSemanticDecoder(
            input_channels=input_channels,
            decoder_channels=decoder_channels,
            fine_classes=31,
            coarse_classes=16 if coarse_head else None,
        )
        lowest_channels = input_channels[-1]
        self.visibility_head = nn.Sequential(
            nn.LayerNorm(lowest_channels),
            nn.Linear(lowest_channels, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 14),
        )

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        original_size = image.shape[-2:]
        encoded = self.encoder(image)
        features = encoded["features"]
        padded_size = tuple(int(x) for x in encoded.get("padded_size", original_size))
        output = self.decoder(features, padded_size)
        pad_h, pad_w = encoded.get("padding", (0, 0))
        if pad_h or pad_w:
            output = {key: value[..., : original_size[0], : original_size[1]] for key, value in output.items()}
        pooled = features[-1].mean(dim=(-2, -1))
        output["visibility_logits"] = self.visibility_head(pooled)
        return output

    def parameter_groups(self, encoder_lr: float, decoder_lr: float, visibility_lr: float) -> list[dict]:
        encoder_parameters = [p for p in self.encoder.parameters() if p.requires_grad]
        decoder_parameters = [p for p in self.decoder.parameters() if p.requires_grad]
        visibility_parameters = [p for p in self.visibility_head.parameters() if p.requires_grad]
        if not decoder_parameters or not visibility_parameters:
            raise RuntimeError("Decoder and visibility parameter groups must be non-empty")
        groups = []
        if encoder_parameters:
            groups.append({"name": "encoder", "params": encoder_parameters, "lr": encoder_lr})
        groups.extend([
            {"name": "decoder", "params": decoder_parameters, "lr": decoder_lr},
            {"name": "visibility_head", "params": visibility_parameters, "lr": visibility_lr},
        ])
        return groups

    def parameter_counts(self) -> dict[str, int]:
        return {
            "total": sum(p.numel() for p in self.parameters()),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
