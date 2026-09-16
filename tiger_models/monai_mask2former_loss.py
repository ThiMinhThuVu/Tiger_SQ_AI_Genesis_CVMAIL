"""MONAI auxiliary semantic loss for Mask2Former query outputs."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def semantic_scores(output) -> torch.Tensor:
    """Combine class queries and mask queries into differentiable class scores."""
    class_probabilities = output.class_queries_logits.softmax(dim=-1)[..., :-1]
    mask_probabilities = output.masks_queries_logits.sigmoid()
    return torch.einsum("bqc,bqhw->bchw", class_probabilities, mask_probabilities)


def build_monai_dice_focal_loss():
    """Construct the configured MONAI loss lazily for clearer dependency errors."""
    try:
        from monai.losses import DiceFocalLoss
    except ImportError as exc:
        raise RuntimeError(
            "MONAI loss is enabled, but MONAI is unavailable; install monai==1.5.2"
        ) from exc
    return DiceFocalLoss(
        include_background=True,
        to_onehot_y=True,
        softmax=True,
        gamma=2.0,
        lambda_dice=1.0,
        lambda_focal=1.0,
        reduction="mean",
    )


def mask2former_monai_loss(output, target: torch.Tensor, criterion) -> torch.Tensor:
    """Compute Dice-Focal on semantic scores at the decoder mask resolution."""
    scores = semantic_scores(output).float()
    resized_target = F.interpolate(
        target[:, None].float(), size=scores.shape[-2:], mode="nearest"
    ).long()
    # DiceFocalLoss expects logits when softmax=True. Log normalized scores are
    # equivalent logits because softmax(log(p)) recovers the normalized p.
    probabilities = scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-6)
    semantic_logits = probabilities.clamp_min(1e-6).log()
    return criterion(semantic_logits, resized_target)
