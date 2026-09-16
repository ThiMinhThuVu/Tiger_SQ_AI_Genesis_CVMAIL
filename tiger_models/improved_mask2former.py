"""Leakage-safe Mask2Former improvements for TIGER surgical segmentation.

This module is intentionally separate from the baseline implementation.  It
contains the train-time method components; experiment runners decide which
components are enabled for an ablation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn
from torch.nn import functional as F


def semantic_probabilities(class_logits: Tensor, mask_logits: Tensor) -> Tensor:
    classes = class_logits.softmax(-1)[..., :-1]
    masks = mask_logits.sigmoid()
    probabilities = torch.einsum("bqc,bqhw->bchw", classes, masks)
    return probabilities / probabilities.sum(1, keepdim=True).clamp_min(1e-6)


def aggregate_fine_to_coarse(fine_probability: Tensor, fine_to_coarse: Tensor) -> Tensor:
    matrix = F.one_hot(fine_to_coarse.long(), num_classes=16).T.to(fine_probability)
    return torch.einsum("cf,bfhw->bchw", matrix, fine_probability)


def symmetric_kl(first: Tensor, second: Tensor) -> Tensor:
    first = first.clamp_min(1e-6); second = second.clamp_min(1e-6)
    return 0.5 * ((first * (first.log() - second.log())).sum(1).mean()
                  + (second * (second.log() - first.log())).sum(1).mean())


def soft_presence_modulation(probability: Tensor, presence: Tensor, epsilon: float, gamma: float) -> Tensor:
    scale = (epsilon + (1.0 - epsilon) * presence.sigmoid()).pow(gamma)
    output = probability * scale[:, :, None, None]
    return output / output.sum(1, keepdim=True).clamp_min(1e-6)


class ClinicalHungarianMatcher(nn.Module):
    """Hungarian assignment with the target clinical weight on every cost."""

    def __init__(self, class_weights: Tensor, cost_class: float, cost_mask: float,
                 cost_dice: float, num_points: int) -> None:
        super().__init__()
        self.register_buffer("class_weights", class_weights.float())
        self.cost_class, self.cost_mask, self.cost_dice = cost_class, cost_mask, cost_dice
        self.num_points = int(num_points)

    @torch.no_grad()
    def forward(self, masks_queries_logits: Tensor, class_queries_logits: Tensor,
                mask_labels: list[Tensor], class_labels: list[Tensor]):
        from transformers.models.mask2former.modeling_mask2former import (
            pair_wise_dice_loss, pair_wise_sigmoid_cross_entropy_loss, sample_point,
        )
        result = []
        for pred_mask, pred_class, target_mask, target_class in zip(
                masks_queries_logits, class_queries_logits, mask_labels, class_labels):
            target_mask = target_mask.to(pred_mask)[:, None]
            coords = torch.rand(1, self.num_points, 2, device=pred_mask.device)
            sampled_target = sample_point(target_mask, coords.repeat(len(target_mask), 1, 1),
                                          align_corners=False).squeeze(1)
            sampled_pred = sample_point(pred_mask[:, None], coords.repeat(len(pred_mask), 1, 1),
                                        align_corners=False).squeeze(1)
            cls = -pred_class.softmax(-1)[:, target_class]
            mask = pair_wise_sigmoid_cross_entropy_loss(sampled_pred, sampled_target)
            dice = pair_wise_dice_loss(sampled_pred, sampled_target)
            weight = self.class_weights[target_class].to(cls)[None]
            cost = weight * (self.cost_class * cls + self.cost_mask * mask + self.cost_dice * dice)
            rows, cols = linear_sum_assignment(torch.nan_to_num(cost, nan=0., posinf=1e10,
                                                                 neginf=-1e10).cpu())
            result.append((torch.as_tensor(rows, dtype=torch.long), torch.as_tensor(cols, dtype=torch.long)))
        return result


class ClinicalMask2FormerLoss(nn.Module):
    """Mask2Former set loss with clinical weights in matching and all terms."""

    def __init__(self, config, class_weights: Tensor) -> None:
        super().__init__()
        self.num_labels = int(config.num_labels)
        self.num_points = int(config.train_num_points)
        self.oversample_ratio = float(config.oversample_ratio)
        self.importance_sample_ratio = float(config.importance_sample_ratio)
        self.register_buffer("class_weights", class_weights.float())
        ce_weights = torch.cat([class_weights.float(), torch.tensor([config.no_object_weight])])
        self.register_buffer("ce_weights", ce_weights)
        self.matcher = ClinicalHungarianMatcher(class_weights, config.class_weight,
                                                 config.mask_weight, config.dice_weight,
                                                 config.train_num_points)
        self.weight_dict = {"loss_cross_entropy": config.class_weight,
                            "loss_mask": config.mask_weight, "loss_dice": config.dice_weight}

    @staticmethod
    def _prediction_indices(indices):
        return (torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)]),
                torch.cat([src for src, _ in indices]))

    def _one_level(self, masks: Tensor, classes: Tensor, mask_labels: list[Tensor],
                   class_labels: list[Tensor]) -> dict[str, Tensor]:
        from transformers.models.mask2former.modeling_mask2former import sample_point
        indices = self.matcher(masks, classes, mask_labels, class_labels)
        pred_idx = self._prediction_indices(indices)
        matched_classes = torch.cat([labels[j] for labels, (_, j) in zip(class_labels, indices)])
        targets = torch.full(classes.shape[:2], self.num_labels, dtype=torch.long, device=classes.device)
        targets[pred_idx] = matched_classes
        loss_ce = F.cross_entropy(classes.transpose(1, 2), targets, weight=self.ce_weights)
        pred_masks = masks[pred_idx][:, None]
        target_masks = torch.cat([labels[j] for labels, (_, j) in zip(mask_labels, indices)])[:, None].to(pred_masks)
        coords = torch.rand(len(pred_masks), self.num_points, 2, device=pred_masks.device)
        pred_points = sample_point(pred_masks, coords, align_corners=False).squeeze(1)
        target_points = sample_point(target_masks, coords, align_corners=False).squeeze(1)
        weights = self.class_weights[matched_classes].to(pred_points)
        bce = F.binary_cross_entropy_with_logits(pred_points, target_points, reduction="none").mean(1)
        probability = pred_points.sigmoid()
        dice = 1.0 - (2 * (probability * target_points).sum(1) + 1) / (
            probability.sum(1) + target_points.sum(1) + 1)
        denominator = weights.sum().clamp_min(1.)
        return {"loss_cross_entropy": loss_ce, "loss_mask": (bce * weights).sum() / denominator,
                "loss_dice": (dice * weights).sum() / denominator}

    def forward(self, masks_queries_logits: Tensor, class_queries_logits: Tensor,
                mask_labels: list[Tensor], class_labels: list[Tensor],
                auxiliary_predictions=None) -> dict[str, Tensor]:
        losses = self._one_level(masks_queries_logits, class_queries_logits, mask_labels, class_labels)
        if auxiliary_predictions:
            for level, prediction in enumerate(auxiliary_predictions):
                extra = self._one_level(prediction["masks_queries_logits"],
                                        prediction["class_queries_logits"], mask_labels, class_labels)
                losses.update({f"{key}_{level}": value for key, value in extra.items()})
        return losses


class ImprovedMask2Former(nn.Module):
    """Stock fine decoder plus auxiliary anatomy and surgical-context heads."""

    def __init__(self, segmenter: nn.Module, fine_to_coarse: Tensor, class_weights: Tensor,
                 matching_method: str = "clinical_hungarian", ot_options: dict | None = None,
                 station_class_prior: Tensor | None = None,
                 context_samples: int = 1, context_dropout: float = 0.0) -> None:
        super().__init__()
        self.segmenter = segmenter
        channels = int(segmenter.config.feature_size)
        self.coarse_head = nn.Conv2d(channels, 16, 1)
        self.presence_head = nn.Linear(channels, 31)
        self.station_head = nn.Linear(channels, 14) if station_class_prior is not None else None
        self.context_samples = int(context_samples)
        self.context_dropout = float(context_dropout)
        if self.context_samples < 1:
            raise ValueError("context_samples must be positive")
        if not 0.0 <= self.context_dropout < 1.0:
            raise ValueError("context_dropout must be in [0, 1)")
        self.register_buffer("fine_to_coarse", fine_to_coarse.long())
        if matching_method == "clinical_hungarian":
            criterion = ClinicalMask2FormerLoss(segmenter.config, class_weights)
        elif matching_method == "risk_sinkhorn":
            from tiger_models.risk_ot_mask2former import RiskSinkhornMask2FormerLoss
            criterion = RiskSinkhornMask2FormerLoss(segmenter.config, class_weights, **(ot_options or {}))
        elif matching_method == "asymmetric_partial_sinkhorn":
            from tiger_models.asymmetric_partial_ot_mask2former import (
                AsymmetricPartialOTMask2FormerLoss,
            )
            criterion = AsymmetricPartialOTMask2FormerLoss(
                segmenter.config, class_weights, station_class_prior=station_class_prior,
                **(ot_options or {}))
        else:
            raise ValueError(f"Unknown matching_method: {matching_method}")
        segmenter.criterion = criterion
        segmenter.weight_dict = segmenter.criterion.weight_dict

    def forward(self, pixel_values: Tensor, mask_labels=None, class_labels=None):
        context_conditioned = self.station_head is not None and mask_labels is not None and class_labels is not None
        output = self.segmenter(
            pixel_values=pixel_values,
            mask_labels=None if context_conditioned else mask_labels,
            class_labels=None if context_conditioned else class_labels,
            output_auxiliary_logits=True if context_conditioned else None,
        )
        feature = output.pixel_decoder_last_hidden_state
        pooled = feature.mean((-2, -1))
        station_logits = self.station_head(pooled) if self.station_head is not None else None
        if context_conditioned:
            loss_dict = self.segmenter.criterion(
                masks_queries_logits=output.masks_queries_logits,
                class_queries_logits=output.class_queries_logits,
                mask_labels=mask_labels,
                class_labels=class_labels,
                auxiliary_predictions=output.auxiliary_logits,
                context_probability=self._context_probabilities(pooled, station_logits),
            )
            for key, weight in self.segmenter.weight_dict.items():
                for loss_key, loss in loss_dict.items():
                    if key in loss_key:
                        loss.mul_(weight)
            output.loss = sum(loss_dict.values())
        fine = semantic_probabilities(output.class_queries_logits, output.masks_queries_logits)
        return {"output": output, "fine_probability": fine,
                "coarse_logits": self.coarse_head(feature),
                "presence_logits": self.presence_head(feature.mean((-2, -1))),
                "station_logits": station_logits, "feature": feature}

    def _context_probabilities(self, pooled: Tensor, station_logits: Tensor) -> Tensor:
        """Create detached context samples; sample zero is unperturbed."""
        probabilities = [station_logits.sigmoid()]
        for _ in range(self.context_samples - 1):
            perturbed = F.dropout(pooled, p=self.context_dropout, training=True)
            probabilities.append(self.station_head(perturbed).sigmoid())
        return torch.stack(probabilities, dim=1).detach()


def presence_targets(fine_target: Tensor, classes: int = 31) -> Tensor:
    return torch.stack([torch.bincount(item.flatten(), minlength=classes).gt(0) for item in fine_target]).float()


def presence_pos_weight(targets: Tensor, maximum: float = 8.) -> Tensor:
    positives = targets.sum(0); negatives = len(targets) - positives
    return (negatives / positives.clamp_min(1)).clamp(1, maximum)


@dataclass
class ObjectiveWeights:
    fine: float = 1.0
    coarse: float = 0.6
    hierarchy: float = 0.2
    presence: float = 0.2
