from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


ROOT = Path(__file__).resolve().parents[1]
SAM2_ROOT = ROOT / "external" / "sam2"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
EXPECTED_CHECKPOINT_SHA256 = "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69"


class SAM2HieraEncoder(nn.Module):
    """Prompt-free SAM 2.1 Hiera encoder exposing all neck scales."""

    feature_channels = (256, 256, 256, 256)

    def __init__(
        self,
        checkpoint_path: str | Path,
        finetune: str = "partial",
        gradient_checkpointing: bool = True,
        config_name: str = SAM2_CONFIG,
    ) -> None:
        super().__init__()
        if not SAM2_ROOT.is_dir():
            raise FileNotFoundError(
                f"Official SAM 2 checkout missing at {SAM2_ROOT}; do not substitute another encoder"
            )
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"SAM 2.1 Hiera checkpoint missing: {checkpoint_path}. "
                "Download the exact registered checkpoint before training."
            )
        sys.path.insert(0, str(SAM2_ROOT)) if str(SAM2_ROOT) not in sys.path else None
        from sam2.build_sam import build_sam2

        # build_sam2 performs a strict missing/unexpected-key compatibility check.
        full_model = build_sam2(
            config_name,
            ckpt_path=str(checkpoint_path),
            device="cpu",
            mode="train",
            apply_postprocessing=False,
        )
        self.image_encoder = full_model.image_encoder
        del full_model
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.finetune = finetune
        self.configure_finetuning(finetune)

    @property
    def trunk(self) -> nn.Module:
        return self.image_encoder.trunk

    @property
    def neck(self) -> nn.Module:
        return self.image_encoder.neck

    def configure_finetuning(self, mode: str) -> None:
        if mode not in {"frozen", "partial", "full"}:
            raise ValueError("finetune must be one of: frozen, partial, full")
        for parameter in self.image_encoder.parameters():
            parameter.requires_grad = mode == "full"
        if mode == "partial":
            for parameter in self.neck.parameters():
                parameter.requires_grad = True
            # Adapt the final two blocks of each Hiera variant.
            if len(self.trunk.blocks) < 2:
                raise RuntimeError("Hiera trunk must contain at least two blocks")
            for block_index in range(len(self.trunk.blocks) - 2, len(self.trunk.blocks)):
                for parameter in self.trunk.blocks[block_index].parameters():
                    parameter.requires_grad = True
        self.finetune = mode

    def freeze_policy(self) -> dict[str, object]:
        trainable_blocks = [
            index for index, block in enumerate(self.trunk.blocks)
            if any(parameter.requires_grad for parameter in block.parameters())
        ]
        return {
            "mode": self.finetune,
            "trainable_trunk_blocks": trainable_blocks,
            "trainable_neck": any(p.requires_grad for p in self.neck.parameters()),
            "frozen_patch_embed": not any(p.requires_grad for p in self.trunk.patch_embed.parameters()),
            "gradient_checkpointing": self.gradient_checkpointing,
            "consumed_neck_levels": 4,
            "upstream_image_encoder_scalp": int(self.image_encoder.scalp),
        }

    @staticmethod
    def _pad_input(image: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        height, width = image.shape[-2:]
        pad_h = (-height) % 32
        pad_w = (-width) % 32
        if pad_h or pad_w:
            image = F.pad(image, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
        return image, (pad_h, pad_w)

    @staticmethod
    def _module_trainable(module: nn.Module) -> bool:
        return any(parameter.requires_grad for parameter in module.parameters())

    def _trunk_forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        trunk = self.trunk
        patch_trainable = self._module_trainable(trunk.patch_embed) or trunk.pos_embed.requires_grad
        if patch_trainable:
            x = trunk.patch_embed(image)
            x = x + trunk._get_pos_embed(x.shape[1:3])
        else:
            with torch.no_grad():
                x = trunk.patch_embed(image)
                x = x + trunk._get_pos_embed(x.shape[1:3])

        outputs: list[torch.Tensor] = []
        stage_ends = set(trunk.stage_ends)
        for index, block in enumerate(trunk.blocks):
            trainable = self._module_trainable(block)
            if trainable and self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            elif trainable or x.requires_grad:
                x = block(x)
            else:
                with torch.no_grad():
                    x = block(x)
            if index in stage_ends:
                outputs.append(x.permute(0, 3, 1, 2))
        if len(outputs) != 4:
            raise RuntimeError(f"Expected four Hiera stages, got {len(outputs)}")
        return outputs

    def forward(self, image: torch.Tensor) -> dict[str, object]:
        image, padding = self._pad_input(image)
        trunk_features = self._trunk_forward(image)
        if self._module_trainable(self.neck) or any(x.requires_grad for x in trunk_features):
            features, _ = self.neck(trunk_features)
        else:
            with torch.no_grad():
                features, _ = self.neck(trunk_features)
        return {
            "features": features,
            "padding": padding,
            "padded_size": image.shape[-2:],
        }


SAM2HieraTinyEncoder = SAM2HieraEncoder
