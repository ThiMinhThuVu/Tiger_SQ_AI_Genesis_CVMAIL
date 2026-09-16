"""Cross-entropy Tversky auxiliary loss for dense Mask2Former outputs."""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def cross_entropy_tversky_loss(
    probabilities: Tensor,
    target: Tensor,
    *,
    class_weights: Tensor | None = None,
    ce_fraction: float = 0.5,
    alpha: float = 0.3,
    beta: float = 0.7,
    include_background: bool = True,
    present_classes_only: bool = True,
    smooth: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Return CETL and its CE/Tversky components.

    ``probabilities`` are the normalized dense class probabilities produced by
    Mask2Former. Targets are resized with nearest-neighbour interpolation to the
    decoder mask grid. When ``present_classes_only`` is enabled, classes absent
    from the current batch do not enter the Tversky average; cross-entropy still
    penalizes false probability mass assigned to them.
    """
    if probabilities.ndim != 4:
        raise ValueError("probabilities must have shape [batch, classes, height, width]")
    if target.ndim != 3:
        raise ValueError("target must have shape [batch, height, width]")
    if probabilities.shape[0] != target.shape[0]:
        raise ValueError("probabilities and target must have the same batch size")
    if not 0.0 <= ce_fraction <= 1.0:
        raise ValueError("ce_fraction must be in [0, 1]")
    if alpha < 0.0 or beta < 0.0 or alpha + beta <= 0.0:
        raise ValueError("alpha and beta must be non-negative with a positive sum")
    if smooth <= 0.0:
        raise ValueError("smooth must be positive")

    classes = probabilities.shape[1]
    resized_target = F.interpolate(
        target[:, None].float(), size=probabilities.shape[-2:], mode="nearest"
    )[:, 0].long()
    if resized_target.numel() and (
        int(resized_target.min()) < 0 or int(resized_target.max()) >= classes
    ):
        raise ValueError("target contains a class outside the probability channels")

    probabilities = probabilities.float()
    probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-6)
    log_probabilities = probabilities.clamp_min(1e-6).log()
    weights = None if class_weights is None else class_weights.to(probabilities).float()
    if weights is not None and weights.numel() != classes:
        raise ValueError(f"class_weights has {weights.numel()} values, expected {classes}")
    cross_entropy = F.nll_loss(log_probabilities, resized_target, weight=weights)

    one_hot = F.one_hot(resized_target, num_classes=classes).permute(0, 3, 1, 2)
    one_hot = one_hot.to(probabilities)
    reduction_dims = (0, 2, 3)
    true_positive = (probabilities * one_hot).sum(reduction_dims)
    false_positive = (probabilities * (1.0 - one_hot)).sum(reduction_dims)
    false_negative = ((1.0 - probabilities) * one_hot).sum(reduction_dims)
    tversky_index = (true_positive + smooth) / (
        true_positive + alpha * false_positive + beta * false_negative + smooth
    )

    valid = torch.ones(classes, dtype=torch.bool, device=probabilities.device)
    if present_classes_only:
        valid &= one_hot.sum(reduction_dims) > 0
    if not include_background:
        valid[0] = False
    if not bool(valid.any()):
        tversky = probabilities.sum() * 0.0
    elif weights is None:
        tversky = 1.0 - tversky_index[valid].mean()
    else:
        valid_weights = weights[valid]
        tversky = 1.0 - (
            tversky_index[valid] * valid_weights
        ).sum() / valid_weights.sum().clamp_min(1e-6)

    total = ce_fraction * cross_entropy + (1.0 - ce_fraction) * tversky
    return total, {"cross_entropy": cross_entropy, "tversky": tversky}
