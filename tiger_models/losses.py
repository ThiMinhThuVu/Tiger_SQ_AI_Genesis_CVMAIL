from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PrimarySAMLoss(nn.Module):
    def __init__(
        self,
        fine_weights: torch.Tensor,
        visibility_pos_weight: torch.Tensor,
        boundary_weight: float = 0.20,
    ) -> None:
        super().__init__()
        if fine_weights.numel() != 31 or visibility_pos_weight.numel() != 14:
            raise ValueError("Expected 31 fine weights and 14 visibility pos_weights")
        self.register_buffer("fine_weights", fine_weights.float())
        self.register_buffer("visibility_pos_weight", visibility_pos_weight.float())
        self.boundary_weight = float(boundary_weight)

    def fine_soft_dice(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probabilities = logits.softmax(dim=1)
        one_hot = F.one_hot(target, 31).permute(0, 3, 1, 2).to(probabilities.dtype)
        dims = (0, 2, 3)
        intersection = (probabilities * one_hot).sum(dims)
        denominator = probabilities.sum(dims) + one_hot.sum(dims)
        dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
        return 1.0 - (dice * self.fine_weights).sum() / self.fine_weights.sum()

    @staticmethod
    def boundary_loss(logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Semantic boundary BCE + Dice on horizontal and vertical pixel edges."""
        probabilities = logits.softmax(dim=1)
        predicted_edges = (
            1.0 - (probabilities[:, :, :, 1:] * probabilities[:, :, :, :-1]).sum(1),
            1.0 - (probabilities[:, :, 1:, :] * probabilities[:, :, :-1, :]).sum(1),
        )
        target_edges = (
            (target[:, :, 1:] != target[:, :, :-1]).to(probabilities.dtype),
            (target[:, 1:, :] != target[:, :-1, :]).to(probabilities.dtype),
        )
        bce_terms = []
        intersections = logits.new_zeros(())
        denominators = logits.new_zeros(())
        for prediction, truth in zip(predicted_edges, target_edges):
            prediction = prediction.clamp(1e-5, 1.0 - 1e-5)
            positives = truth.sum()
            negatives = truth.numel() - positives
            positive_weight = (negatives / positives.clamp_min(1.0)).clamp(1.0, 20.0)
            bce_terms.append(
                -(positive_weight * truth * prediction.log()
                  + (1.0 - truth) * (1.0 - prediction).log()).mean()
            )
            intersections = intersections + (prediction * truth).sum()
            denominators = denominators + prediction.sum() + truth.sum()
        boundary_bce = torch.stack(bce_terms).mean()
        boundary_dice = 1.0 - (2.0 * intersections + 1.0) / (denominators + 1.0)
        return boundary_bce, boundary_dice

    def forward(
        self,
        output: dict[str, torch.Tensor],
        fine_target: torch.Tensor,
        visibility_target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        fine_logits = output["fine_logits"]
        visibility_logits = output["visibility_logits"]
        fine_ce = F.cross_entropy(fine_logits, fine_target, weight=self.fine_weights)
        fine_dice = self.fine_soft_dice(fine_logits, fine_target)
        boundary_bce, boundary_dice = self.boundary_loss(fine_logits, fine_target)
        visibility_bce = F.binary_cross_entropy_with_logits(
            visibility_logits, visibility_target, pos_weight=self.visibility_pos_weight
        )
        boundary = boundary_bce + boundary_dice
        total = 0.50 * fine_ce + fine_dice + 0.40 * visibility_bce + self.boundary_weight * boundary
        return {
            "loss": total,
            "fine_ce": fine_ce,
            "fine_soft_dice": fine_dice,
            "boundary_bce": boundary_bce,
            "boundary_dice": boundary_dice,
            "boundary_loss": boundary,
            "visibility_bce": visibility_bce,
        }
