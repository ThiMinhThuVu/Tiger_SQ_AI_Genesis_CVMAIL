"""Risk-conditioned entropic optimal-transport loss for Mask2Former."""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def log_sinkhorn(cost: Tensor, row_mass: Tensor, column_mass: Tensor,
                 epsilon: float, iterations: int) -> Tensor:
    """Solve balanced entropic OT in log space for one cost matrix."""
    if cost.ndim != 2 or row_mass.ndim != 1 or column_mass.ndim != 1:
        raise ValueError("Expected cost [Q,T], row_mass [Q], column_mass [T]")
    if cost.shape != (row_mass.numel(), column_mass.numel()):
        raise ValueError("OT cost and marginal shapes differ")
    if epsilon <= 0 or iterations < 1:
        raise ValueError("epsilon and iterations must be positive")
    output_dtype = cost.dtype
    # OT is solved in float64: clinical weighting and small epsilon can create
    # log-kernel ranges where float32 exponentiation destroys feasible edges.
    cost = cost.double(); row_mass = row_mass.double(); column_mass = column_mass.double()
    log_kernel = -cost / epsilon
    log_row = row_mass.clamp_min(1e-300).log()
    log_column = column_mass.clamp_min(1e-300).log()
    log_u = torch.zeros_like(log_row); log_v = torch.zeros_like(log_column)
    for _ in range(iterations):
        log_u = log_row - torch.logsumexp(log_kernel + log_v[None, :], dim=1)
        log_v = log_column - torch.logsumexp(log_kernel + log_u[:, None], dim=0)
    plan = torch.exp(log_kernel + log_u[:, None] + log_v[None, :]).clamp_min(1e-300)
    # A short probability-domain balancing pass removes the residual marginal
    # error left by very sharp, high-clinical-risk costs. Assignment is used
    # under no_grad, so this stabilization adds no backward graph or sync.
    for _ in range(100):
        plan = plan * (row_mass / plan.sum(1).clamp_min(1e-300))[:, None]
        plan = plan * (column_mass / plan.sum(0).clamp_min(1e-300))[None, :]
    return plan.to(output_dtype)


class RiskSinkhornMask2FormerLoss(nn.Module):
    """Expected set loss under a clinical risk-conditioned OT assignment.

    Each real target receives mass 1/Q. Remaining mass is assigned to an
    explicit no-object dustbin, so every query has row mass 1/Q.
    """

    def __init__(self, config, class_weights: Tensor, epsilon: float = 0.08,
                 iterations: int = 40, risk_exponent: float = 1.0,
                 dustbin_cost: float = 0.0) -> None:
        super().__init__()
        self.num_labels = int(config.num_labels)
        self.num_points = int(config.train_num_points)
        self.epsilon = float(epsilon); self.iterations = int(iterations)
        self.risk_exponent = float(risk_exponent); self.dustbin_cost = float(dustbin_cost)
        self.cost_class = float(config.class_weight); self.cost_mask = float(config.mask_weight)
        self.cost_dice = float(config.dice_weight); self.no_object_weight = float(config.no_object_weight)
        self.register_buffer("class_weights", class_weights.float())
        self.weight_dict = {"loss_cross_entropy": self.cost_class,
                            "loss_mask": self.cost_mask, "loss_dice": self.cost_dice}
        self.last_diagnostics: dict[str, float] = {}

    def _pairwise(self, masks: Tensor, classes: Tensor, target_masks: Tensor,
                  target_classes: Tensor):
        from transformers.models.mask2former.modeling_mask2former import (
            pair_wise_dice_loss, pair_wise_sigmoid_cross_entropy_loss, sample_point,
        )
        coordinates = torch.rand(1, self.num_points, 2, device=masks.device)
        pred_points = sample_point(masks[:, None], coordinates.repeat(len(masks), 1, 1),
                                   align_corners=False).squeeze(1)
        target_points = sample_point(target_masks[:, None].to(masks),
                                     coordinates.repeat(len(target_masks), 1, 1),
                                     align_corners=False).squeeze(1)
        class_cost = -classes.softmax(-1)[:, target_classes]
        mask_cost = pair_wise_sigmoid_cross_entropy_loss(pred_points, target_points)
        dice_cost = pair_wise_dice_loss(pred_points, target_points)
        return pred_points, target_points, class_cost, mask_cost, dice_cost

    def _one_image(self, masks: Tensor, classes: Tensor, target_masks: Tensor,
                   target_classes: Tensor):
        q, targets = len(masks), len(target_classes)
        if targets == 0 or targets > q:
            raise ValueError(f"Risk OT expects 1..Q targets, got Q={q}, T={targets}")
        pred_points, target_points, cls_cost, mask_cost, dice_cost = self._pairwise(
            masks, classes, target_masks, target_classes)
        risk = self.class_weights[target_classes].to(masks).pow(self.risk_exponent)
        real_cost = risk[None, :] * (self.cost_class * cls_cost + self.cost_mask * mask_cost
                                     + self.cost_dice * dice_cost)
        dustbin = masks.new_full((q, 1), self.dustbin_cost)
        cost = torch.cat([real_cost, dustbin], dim=1)
        row_mass = masks.new_full((q,), 1.0 / q)
        column_mass = masks.new_full((targets + 1,), 1.0 / q)
        column_mass[-1] = (q - targets) / q
        with torch.no_grad():
            plan = log_sinkhorn(cost.detach(), row_mass, column_mass,
                                self.epsilon, self.iterations)
        # Normalize every query row into a soft class target distribution.
        soft = classes.new_zeros((q, self.num_labels + 1))
        soft[:, target_classes] += plan[:, :targets] / row_mass[:, None]
        soft[:, -1] = plan[:, -1] / row_mass
        log_probability = classes.log_softmax(-1)
        class_scale = torch.cat([self.class_weights.to(classes).pow(self.risk_exponent),
                                 classes.new_tensor([self.no_object_weight])])
        classification = -(soft * log_probability * class_scale[None, :]).sum(1).mean()

        # Expected pair loss under real-target transport mass.
        pair_bce = F.binary_cross_entropy_with_logits(
            pred_points[:, None, :].expand(-1, targets, -1),
            target_points[None, :, :].expand(q, -1, -1), reduction="none").mean(-1)
        probability = pred_points.sigmoid()
        intersection = torch.einsum("qp,tp->qt", probability, target_points)
        pair_dice = 1.0 - (2 * intersection + 1) / (
            probability.sum(1)[:, None] + target_points.sum(1)[None, :] + 1)
        weighted_plan = plan[:, :targets] * risk[None, :]
        normalizer = weighted_plan.sum().clamp_min(1e-8)
        mask_loss = (weighted_plan * pair_bce).sum() / normalizer
        dice_loss = (weighted_plan * pair_dice).sum() / normalizer
        entropy = -(plan.clamp_min(1e-12) * plan.clamp_min(1e-12).log()).sum()
        diagnostics = {"transport_entropy": float(entropy.detach()),
                       "dustbin_mass": float(plan[:, -1].sum().detach()),
                       "marginal_error": float(max((plan.sum(1)-row_mass).abs().max(),
                                                   (plan.sum(0)-column_mass).abs().max()).detach())}
        return classification, mask_loss, dice_loss, diagnostics

    def _one_level(self, masks: Tensor, classes: Tensor, mask_labels: list[Tensor],
                   class_labels: list[Tensor]):
        rows = [self._one_image(pm, pc, tm.to(pm), tc.to(pc.device))
                for pm, pc, tm, tc in zip(masks, classes, mask_labels, class_labels)]
        self.last_diagnostics = {key: sum(row[3][key] for row in rows) / len(rows)
                                 for key in rows[0][3]}
        return {"loss_cross_entropy": torch.stack([row[0] for row in rows]).mean(),
                "loss_mask": torch.stack([row[1] for row in rows]).mean(),
                "loss_dice": torch.stack([row[2] for row in rows]).mean()}

    def forward(self, masks_queries_logits: Tensor, class_queries_logits: Tensor,
                mask_labels: list[Tensor], class_labels: list[Tensor],
                auxiliary_predictions=None):
        losses = self._one_level(masks_queries_logits, class_queries_logits,
                                 mask_labels, class_labels)
        if auxiliary_predictions:
            for level, prediction in enumerate(auxiliary_predictions):
                extra = self._one_level(prediction["masks_queries_logits"],
                                        prediction["class_queries_logits"],
                                        mask_labels, class_labels)
                losses.update({f"{key}_{level}": value for key, value in extra.items()})
        return losses
