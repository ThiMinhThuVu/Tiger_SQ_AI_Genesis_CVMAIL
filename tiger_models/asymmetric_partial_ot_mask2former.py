"""Presence-conditioned asymmetric partial-OT loss for Mask2Former.

PC-APOT separates false-negative and false-positive clinical risk. Present
targets receive risk-dependent transport mass, while dustbin-assigned queries
receive an explicit risk-weighted foreground suppression loss. The existing A2
balanced risk-OT implementation is intentionally left unchanged.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from tiger_models.risk_ot_mask2former import log_sinkhorn


def log_sinkhorn_converged(cost: Tensor, row_mass: Tensor, column_mass: Tensor,
                           epsilon: float, max_iterations: int = 500,
                           tolerance: float = 1e-7, check_interval: int = 10,
                           balance_iterations: int = 500) -> tuple[Tensor, dict[str, float]]:
    """Solve entropic OT with an explicit marginal-convergence criterion."""
    if cost.ndim != 2 or cost.shape != (row_mass.numel(), column_mass.numel()):
        raise ValueError("OT cost and marginal shapes differ")
    if epsilon <= 0 or max_iterations < 1 or tolerance <= 0 or check_interval < 1:
        raise ValueError("Invalid convergence-based Sinkhorn parameters")
    output_dtype = cost.dtype
    cost = cost.double(); row_mass = row_mass.double(); column_mass = column_mass.double()
    log_kernel = -cost / epsilon
    log_row = row_mass.clamp_min(1e-300).log()
    log_column = column_mass.clamp_min(1e-300).log()
    log_u = torch.zeros_like(log_row); log_v = torch.zeros_like(log_column)
    log_used = max_iterations
    for iteration in range(1, max_iterations + 1):
        log_u = log_row - torch.logsumexp(log_kernel + log_v[None, :], dim=1)
        log_v = log_column - torch.logsumexp(log_kernel + log_u[:, None], dim=0)
        if iteration % check_interval == 0 or iteration == max_iterations:
            candidate = torch.exp(log_kernel + log_u[:, None] + log_v[None, :])
            error = torch.maximum((candidate.sum(1)-row_mass).abs().max(),
                                  (candidate.sum(0)-column_mass).abs().max())
            if float(error) <= tolerance:
                log_used = iteration
                break
    plan = torch.exp(log_kernel + log_u[:, None] + log_v[None, :]).clamp_min(1e-300)
    balance_used = 0
    for iteration in range(1, balance_iterations + 1):
        plan = plan * (row_mass / plan.sum(1).clamp_min(1e-300))[:, None]
        plan = plan * (column_mass / plan.sum(0).clamp_min(1e-300))[None, :]
        balance_used = iteration
        if iteration % check_interval == 0 or iteration == balance_iterations:
            error = torch.maximum((plan.sum(1)-row_mass).abs().max(),
                                  (plan.sum(0)-column_mass).abs().max())
            if float(error) <= tolerance:
                break
    final_error = torch.maximum((plan.sum(1)-row_mass).abs().max(),
                                (plan.sum(0)-column_mass).abs().max())
    return plan.to(output_dtype), {
        "sinkhorn_iterations": float(log_used),
        "balance_iterations": float(balance_used),
        "solver_marginal_error": float(final_error),
    }


class AsymmetricPartialOTMask2FormerLoss(nn.Module):
    """Expected Mask2Former set loss under PC-APOT assignment."""

    def __init__(self, config, class_weights: Tensor, epsilon: float = 0.5,
                 iterations: int = 120, target_mass_alpha: float = 0.5,
                 target_mass_max: float = 2.0, dustbin_fp_weight: float = 0.1,
                 pair_cost_risk_exponent: float = 0.0,
                 dustbin_cost: float = 0.0, mass_mode: str = "fixed",
                 convergence_tolerance: float = 0.0, check_interval: int = 10,
                 balance_iterations: int = 500,
                 station_class_prior: Tensor | None = None,
                 context_fp_strength: float = 1.0,
                 context_bound_mode: str = "point",
                 sink_margin_matrix: Tensor | None = None,
                 sink_margin_weight: float = 0.0,
                 sink_margin_value: float = 0.05,
                 recall_reweight_vector: Tensor | None = None,
                 recall_reweight_weight: float = 0.0) -> None:
        super().__init__()
        self.num_labels = int(config.num_labels)
        self.num_points = int(config.train_num_points)
        self.epsilon = float(epsilon)
        self.iterations = int(iterations)
        self.target_mass_alpha = float(target_mass_alpha)
        self.target_mass_max = float(target_mass_max)
        self.dustbin_fp_weight = float(dustbin_fp_weight)
        self.pair_cost_risk_exponent = float(pair_cost_risk_exponent)
        self.dustbin_cost = float(dustbin_cost)
        self.mass_mode = str(mass_mode)
        self.convergence_tolerance = float(convergence_tolerance)
        self.check_interval = int(check_interval)
        self.balance_iterations = int(balance_iterations)
        self.context_fp_strength = float(context_fp_strength)
        self.context_bound_mode = str(context_bound_mode)
        self.cost_class = float(config.class_weight)
        self.cost_mask = float(config.mask_weight)
        self.cost_dice = float(config.dice_weight)
        self.no_object_weight = float(config.no_object_weight)
        if self.target_mass_alpha < 0:
            raise ValueError("target_mass_alpha must be non-negative")
        if self.target_mass_max < 1:
            raise ValueError("target_mass_max must be at least one")
        if self.dustbin_fp_weight < 0:
            raise ValueError("dustbin_fp_weight must be non-negative")
        if self.mass_mode not in {"fixed", "adaptive", "context"}:
            raise ValueError("mass_mode must be 'fixed', 'adaptive', or 'context'")
        if self.context_fp_strength < 0:
            raise ValueError("context_fp_strength must be non-negative")
        if self.context_bound_mode not in {"point", "minmax"}:
            raise ValueError("context_bound_mode must be 'point' or 'minmax'")
        self.sink_margin_weight = float(sink_margin_weight)
        self.sink_margin_value = float(sink_margin_value)
        if self.sink_margin_weight < 0:
            raise ValueError("sink_margin_weight must be non-negative")
        if sink_margin_matrix is not None:
            matrix = sink_margin_matrix.float()
            if matrix.shape != (self.num_labels, self.num_labels):
                raise ValueError("sink_margin_matrix must have shape [num_labels, num_labels]")
            self.register_buffer("sink_margin_matrix", matrix)
        else:
            self.sink_margin_matrix = None
        if self.sink_margin_weight > 0 and self.sink_margin_matrix is None:
            raise ValueError("sink_margin_weight > 0 requires sink_margin_matrix")
        self.recall_reweight_weight = float(recall_reweight_weight)
        if self.recall_reweight_weight < 0:
            raise ValueError("recall_reweight_weight must be non-negative")
        if recall_reweight_vector is not None:
            vector = recall_reweight_vector.float()
            if vector.shape != (self.num_labels,):
                raise ValueError("recall_reweight_vector must have shape [num_labels]")
            self.register_buffer("recall_reweight_vector", vector.clamp(0, 1))
        else:
            self.recall_reweight_vector = None
        if self.recall_reweight_weight > 0 and self.recall_reweight_vector is None:
            raise ValueError("recall_reweight_weight > 0 requires recall_reweight_vector")
        self.register_buffer("class_weights", class_weights.float())
        if station_class_prior is not None:
            prior = station_class_prior.float()
            if prior.shape != (14, self.num_labels):
                raise ValueError("station_class_prior must have shape [14, num_labels]")
            prior = prior / prior.amax(0, keepdim=True).clamp_min(1e-6)
            self.register_buffer("station_class_prior", prior.clamp(0, 1))
        else:
            self.station_class_prior = None
        if self.mass_mode == "context" and self.station_class_prior is None:
            raise ValueError("context mass mode requires station_class_prior")
        self.weight_dict = {
            "loss_cross_entropy": self.cost_class,
            "loss_mask": self.cost_mask,
            "loss_dice": self.cost_dice,
            "loss_dustbin_fp": 1.0,
            "loss_sink_margin": 1.0,
        }
        self.last_diagnostics: dict[str, float] = {}
        self.last_class_diagnostics: dict[str, dict[str, float]] = {}

    def target_column_masses(self, target_classes: Tensor, queries: int,
                             reference: Tensor | None = None,
                             mass_gate: Tensor | None = None) -> Tensor:
        """Return present-target masses followed by the residual dustbin mass."""
        if queries < 1 or target_classes.ndim != 1:
            raise ValueError("queries must be positive and target_classes must be one-dimensional")
        if target_classes.numel() == 0 or target_classes.numel() > queries:
            raise ValueError(f"PC-APOT expects 1..Q targets, got Q={queries}, T={target_classes.numel()}")
        device = reference.device if reference is not None else target_classes.device
        dtype = reference.dtype if reference is not None else self.class_weights.dtype
        risk = self.class_weights[target_classes.to(self.class_weights.device)].to(device=device, dtype=dtype)
        if mass_gate is None:
            mass_gate = torch.ones_like(risk)
        else:
            mass_gate = mass_gate.to(device=device, dtype=dtype).clamp(0, 1)
            if mass_gate.shape != risk.shape:
                raise ValueError("mass_gate must have one value per target")
        multiplier = (1.0 + self.target_mass_alpha * (risk - 1.0) * mass_gate).clamp(
            min=1.0, max=self.target_mass_max)
        real_mass = multiplier / float(queries)
        dustbin_mass = real_mass.new_tensor(1.0) - real_mass.sum()
        if not bool(torch.isfinite(dustbin_mass)) or float(dustbin_mass.detach()) <= 0:
            raise ValueError(
                f"Non-positive dustbin mass {float(dustbin_mass.detach()):.6g}; "
                "reduce target_mass_alpha/target_mass_max or increase the query count"
            )
        return torch.cat([real_mass, dustbin_mass[None]])

    def _adaptive_mass_gate(self, base_cost: Tensor, target_masks: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Gate mass by detached assignment ambiguity and relative target size."""
        queries = base_cost.shape[0]
        affinity = (-base_cost.detach().double() / self.epsilon).softmax(dim=0)
        entropy = -(affinity.clamp_min(1e-300) * affinity.clamp_min(1e-300).log()).sum(0)
        ambiguity = (entropy / math.log(max(queries, 2))).clamp(0, 1).to(base_cost)
        area = target_masks.detach().to(base_cost).flatten(1).mean(1).clamp_min(1e-8)
        median_area = area.median().clamp_min(1e-8)
        smallness = (median_area / area).sqrt().clamp(0.5, 1.5)
        gate = (ambiguity * smallness).clamp(0, 1)
        return gate, ambiguity, smallness

    def _context_relevance(self, context_probability: Tensor) -> Tensor:
        if self.station_class_prior is None:
            raise RuntimeError("Context relevance requested without a station-class prior")
        probability = context_probability.to(self.station_class_prior).clamp(0, 1)
        if probability.shape != (14,):
            raise ValueError("context_probability must contain 14 station probabilities")
        return (probability[:, None] * self.station_class_prior).sum(0) / probability.sum().clamp_min(1e-6)

    def _context_relevance_bounds(self, context_probability: Tensor) -> tuple[Tensor, Tensor]:
        """Return conservative anatomy-relevance bounds across context samples."""
        if context_probability.ndim == 1:
            relevance = self._context_relevance(context_probability)
            return relevance, relevance
        if context_probability.ndim != 2 or context_probability.shape[1] != 14:
            raise ValueError("context_probability must have shape [14] or [samples, 14]")
        relevance = torch.stack([
            self._context_relevance(sample) for sample in context_probability
        ])
        if self.context_bound_mode == "point":
            return relevance[0], relevance[0]
        return relevance.amin(0), relevance.amax(0)

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

    def _dustbin_fp_loss(self, classes: Tensor, dustbin_plan: Tensor,
                         dustbin_mass: Tensor,
                         context_relevance: Tensor | None = None) -> tuple[Tensor, Tensor]:
        probabilities = classes.softmax(-1)[:, :self.num_labels]
        normalized_risk = self.class_weights.to(classes)
        normalized_risk = normalized_risk / normalized_risk.mean().clamp_min(1e-8)
        if context_relevance is not None:
            incompatibility = 1.0 - context_relevance.to(classes).clamp(0, 1)
            normalized_risk = normalized_risk * (1.0 + self.context_fp_strength * incompatibility)
        foreground_risk = (probabilities * normalized_risk[None, :]).sum(-1)
        raw = (dustbin_plan * foreground_risk).sum() / dustbin_mass.clamp_min(1e-8)
        return self.dustbin_fp_weight * raw, raw

    def _one_image(self, masks: Tensor, classes: Tensor, target_masks: Tensor,
                   target_classes: Tensor, context_probability: Tensor | None = None):
        queries, targets = len(masks), len(target_classes)
        pred_points, target_points, cls_cost, mask_cost, dice_cost = self._pairwise(
            masks, classes, target_masks, target_classes)
        clinical_risk = self.class_weights[target_classes].to(masks)
        cost_risk = clinical_risk.pow(self.pair_cost_risk_exponent)
        base_cost = (self.cost_class * cls_cost + self.cost_mask * mask_cost
                     + self.cost_dice * dice_cost)
        real_cost = cost_risk[None, :] * base_cost
        dustbin = masks.new_full((queries, 1), self.dustbin_cost)
        cost = torch.cat([real_cost, dustbin], dim=1)
        row_mass = masks.new_full((queries,), 1.0 / queries)
        if context_probability is not None and self.station_class_prior is not None:
            context_relevance_lower, context_relevance_upper = self._context_relevance_bounds(
                context_probability)
        else:
            context_relevance_lower = context_relevance_upper = None
        if self.mass_mode == "adaptive":
            mass_gate, ambiguity, smallness = self._adaptive_mass_gate(base_cost, target_masks)
        elif self.mass_mode == "context":
            if context_relevance_lower is None:
                raise ValueError("context mass mode requires one context posterior per image")
            mass_gate = context_relevance_lower[target_classes].to(base_cost).clamp(0, 1)
            ambiguity = mass_gate.new_zeros(targets)
            smallness = mass_gate.new_ones(targets)
        else:
            mass_gate = masks.new_ones(targets)
            ambiguity = masks.new_zeros(targets)
            smallness = masks.new_ones(targets)
        column_mass = self.target_column_masses(
            target_classes, queries, masks, mass_gate=mass_gate)
        with torch.no_grad():
            if self.convergence_tolerance > 0:
                plan, solver_diagnostics = log_sinkhorn_converged(
                    cost.detach(), row_mass, column_mass, self.epsilon,
                    max_iterations=self.iterations,
                    tolerance=self.convergence_tolerance,
                    check_interval=self.check_interval,
                    balance_iterations=self.balance_iterations)
            else:
                plan = log_sinkhorn(cost.detach(), row_mass, column_mass,
                                    self.epsilon, self.iterations)
                solver_diagnostics = {"sinkhorn_iterations": float(self.iterations),
                                      "balance_iterations": 100.0}

        soft = classes.new_zeros((queries, self.num_labels + 1))
        soft[:, target_classes] += plan[:, :targets] / row_mass[:, None]
        soft[:, -1] = plan[:, -1] / row_mass
        log_probability = classes.log_softmax(-1)
        class_scale = torch.cat([self.class_weights.to(classes),
                                 classes.new_tensor([self.no_object_weight])])
        if self.recall_reweight_weight > 0:
            # Recall Loss (Tian et al., arXiv:2106.14917): boost the loss weight of
            # classes with a high out-of-fold miss rate, no object weight untouched.
            recall_scale = torch.cat([
                1.0 + self.recall_reweight_weight * self.recall_reweight_vector.to(classes),
                classes.new_tensor([1.0]),
            ])
            class_scale = class_scale * recall_scale
        classification = -(soft * log_probability * class_scale[None, :]).sum(1).mean()

        if self.sink_margin_weight > 0:
            # Directed Sink-Margin Loss (DSML): push a matched query's logit for its
            # ground-truth (donor) class above the logit of empirically-confusable
            # sink classes, weighted by an out-of-fold-estimated confusion edge.
            # Acts purely on class_queries_logits -- no interaction with the
            # (no_grad, detached-cost) Sinkhorn plan or matching cost.
            edge = self.sink_margin_matrix.to(classes)[target_classes]  # [T, num_labels]
            donor_logit = classes[:, target_classes]                    # [Q, T]
            sink_logit = classes[:, :self.num_labels]                   # [Q, num_labels]
            diff = donor_logit[:, :, None] - sink_logit[:, None, :]     # [Q, T, num_labels]
            hinge = (self.sink_margin_value - diff).clamp_min(0.0)
            query_weight = soft[:, target_classes]                      # [Q, T]
            weighted = hinge * edge[None, :, :] * query_weight[:, :, None]
            denom = query_weight.sum(0).clamp_min(1e-6)                 # [T]
            per_target = weighted.sum(0).sum(-1) / denom                # [T]
            sink_margin_loss = self.sink_margin_weight * per_target.mean()
        else:
            sink_margin_loss = classes.new_zeros(())

        pair_bce = F.binary_cross_entropy_with_logits(
            pred_points[:, None, :].expand(-1, targets, -1),
            target_points[None, :, :].expand(queries, -1, -1), reduction="none").mean(-1)
        probability = pred_points.sigmoid()
        intersection = torch.einsum("qp,tp->qt", probability, target_points)
        pair_dice = 1.0 - (2 * intersection + 1) / (
            probability.sum(1)[:, None] + target_points.sum(1)[None, :] + 1)
        weighted_plan = plan[:, :targets] * clinical_risk[None, :]
        normalizer = weighted_plan.sum().clamp_min(1e-8)
        mask_loss = (weighted_plan * pair_bce).sum() / normalizer
        dice_loss = (weighted_plan * pair_dice).sum() / normalizer
        fp_loss, raw_fp_risk = self._dustbin_fp_loss(
            classes, plan[:, -1], column_mass[-1], context_relevance_upper)

        entropy = -(plan.clamp_min(1e-12) * plan.clamp_min(1e-12).log()).sum()
        predicted_class = classes.argmax(-1)
        correct_edge = predicted_class[:, None].eq(target_classes[None, :])
        transport_recall = ((plan[:, :targets] * correct_edge).sum(0)
                            / column_mass[:targets].clamp_min(1e-8)).mean()
        marginal_error = torch.maximum((plan.sum(1) - row_mass).abs().max(),
                                       (plan.sum(0) - column_mass).abs().max())
        diagnostics = {
            "transport_entropy": float(entropy.detach()),
            "dustbin_mass": float(plan[:, -1].sum().detach()),
            "marginal_error": float(marginal_error.detach()),
            "target_mass_mean": float(column_mass[:-1].mean().detach()),
            "target_mass_min": float(column_mass[:-1].min().detach()),
            "target_mass_max": float(column_mass[:-1].max().detach()),
            "transport_recall": float(transport_recall.detach()),
            "dustbin_foreground_risk": float(raw_fp_risk.detach()),
            "mass_gate_mean": float(mass_gate.mean().detach()),
            "context_interval_width_mean": float(
                (context_relevance_upper - context_relevance_lower).mean().detach()
                if context_relevance_lower is not None else 0.0),
            "assignment_ambiguity_mean": float(ambiguity.mean().detach()),
            "target_smallness_mean": float(smallness.mean().detach()),
            **solver_diagnostics,
        }
        class_diagnostics = {
            str(int(label)): {
                "requested_mass": float(requested.detach()),
                "transported_mass": float(achieved.detach()),
                "transport_recall": float(recall.detach()),
                "mass_gate": float(gate.detach()),
                "assignment_ambiguity": float(target_ambiguity.detach()),
                "target_smallness": float(target_smallness.detach()),
            }
            for label, requested, achieved, recall, gate, target_ambiguity, target_smallness in zip(
                target_classes.detach().cpu(), column_mass[:-1], plan[:, :targets].sum(0),
                (plan[:, :targets] * correct_edge).sum(0) / column_mass[:targets].clamp_min(1e-8),
                mass_gate, ambiguity, smallness)
        }
        return classification, mask_loss, dice_loss, fp_loss, diagnostics, class_diagnostics, sink_margin_loss

    def _one_level(self, masks: Tensor, classes: Tensor, mask_labels: list[Tensor],
                   class_labels: list[Tensor], context_probability: Tensor | None = None):
        contexts = [None] * len(masks) if context_probability is None else list(context_probability)
        rows = [self._one_image(pm, pc, tm.to(pm), tc.to(pc.device), context)
                for pm, pc, tm, tc, context in zip(
                    masks, classes, mask_labels, class_labels, contexts)]
        self.last_diagnostics = {
            key: sum(row[4][key] for row in rows) / len(rows) for key in rows[0][4]
        }
        merged: dict[str, list[dict[str, float]]] = {}
        for row in rows:
            for label, values in row[5].items():
                merged.setdefault(label, []).append(values)
        self.last_class_diagnostics = {
            label: {key: sum(item[key] for item in values) / len(values)
                    for key in values[0]}
            for label, values in merged.items()
        }
        return {
            "loss_cross_entropy": torch.stack([row[0] for row in rows]).mean(),
            "loss_mask": torch.stack([row[1] for row in rows]).mean(),
            "loss_dice": torch.stack([row[2] for row in rows]).mean(),
            "loss_dustbin_fp": torch.stack([row[3] for row in rows]).mean(),
            "loss_sink_margin": torch.stack([row[6] for row in rows]).mean(),
        }

    def forward(self, masks_queries_logits: Tensor, class_queries_logits: Tensor,
                mask_labels: list[Tensor], class_labels: list[Tensor],
                auxiliary_predictions=None, context_probability: Tensor | None = None):
        losses = self._one_level(masks_queries_logits, class_queries_logits,
                                 mask_labels, class_labels, context_probability)
        main_diagnostics = self.last_diagnostics
        main_class_diagnostics = self.last_class_diagnostics
        if auxiliary_predictions:
            for level, prediction in enumerate(auxiliary_predictions):
                extra = self._one_level(prediction["masks_queries_logits"],
                                        prediction["class_queries_logits"],
                                        mask_labels, class_labels, context_probability)
                losses.update({f"{key}_{level}": value for key, value in extra.items()})
        self.last_diagnostics = main_diagnostics
        self.last_class_diagnostics = main_class_diagnostics
        return losses
