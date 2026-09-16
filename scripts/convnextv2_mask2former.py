#!/usr/bin/env python3
"""ConvNeXt-V2 Base adapter for the repository's HF Mask2Former runner."""
from __future__ import annotations
from types import SimpleNamespace
import torch
import torch.nn as nn


class TimmConvNeXtV2Backbone(nn.Module):
    """Expose timm ConvNeXt-V2 pyramid features as Mask2Former feature_maps."""
    def __init__(self, model_name: str, pretrained: bool = True):
        super().__init__()
        import timm
        self.model = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        self.channels = tuple(self.model.feature_info.channels())
        if hasattr(self.model, "set_grad_checkpointing"):
            self.model.set_grad_checkpointing(True)

    def forward(self, pixel_values: torch.Tensor):
        return SimpleNamespace(feature_maps=self.model(pixel_values))


class ConvNeXtV2Mask2Former(nn.Module):
    """Mask2Former segmentation model plus a 14-label visibility probe.

    The Mask2Former query/mask decoder remains the official Transformers
    implementation; only its pixel-level backbone is replaced by ConvNeXt-V2.
    """
    def __init__(self, model, backbone: TimmConvNeXtV2Backbone):
        super().__init__()
        self.model = model
        self.backbone = backbone
        self.model.model.pixel_level_module.encoder = self.backbone
        self.visibility_head = nn.Sequential(
            nn.LayerNorm(backbone.channels[-1]), nn.Linear(backbone.channels[-1], 256),
            nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 14),
        )

    def forward(self, pixel_values, **kwargs):
        output = self.model(pixel_values=pixel_values, output_hidden_states=True, **kwargs)
        features = output.encoder_last_hidden_state
        visibility_logits = self.visibility_head(features.mean((-2, -1)))
        return output, visibility_logits


def build_model(model_name: str, pretrained: bool, num_labels: int = 31,
                decoder_layers: int = 6, num_queries: int = 100):
    """Build Mask2Former with projections matching Tiny or Base channels."""
    # transformers 5.14 imports an Executorch helper introduced after torch
    # 2.7. The compatibility definition is irrelevant to Mask2Former itself.
    import torch.fx.experimental.symbolic_shapes as symbolic_shapes
    if not hasattr(symbolic_shapes, "guard_or_true"):
        symbolic_shapes.guard_or_true = lambda value: bool(value)
    from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation, SwinConfig

    backbone = TimmConvNeXtV2Backbone(model_name, pretrained=pretrained)
    embed_dim = backbone.channels[0]
    fake_backbone_config = SwinConfig(
        embed_dim=embed_dim, depths=[2, 2, 6, 2],
        num_heads=[4, 8, 16, 32] if embed_dim == 128 else [3, 6, 12, 24],
        out_features=["stage1", "stage2", "stage3", "stage4"],
    )
    config = Mask2FormerConfig(
        backbone_config=fake_backbone_config, num_labels=num_labels,
        decoder_layers=decoder_layers, num_queries=num_queries,
        use_auxiliary_loss=True, output_auxiliary_logits=True,
    )
    segmenter = Mask2FormerForUniversalSegmentation(config)
    return ConvNeXtV2Mask2Former(segmenter, backbone)
