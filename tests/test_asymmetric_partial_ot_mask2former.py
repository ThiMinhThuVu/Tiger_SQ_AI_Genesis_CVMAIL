from __future__ import annotations

import torch

from tiger_models.asymmetric_partial_ot_mask2former import (
    AsymmetricPartialOTMask2FormerLoss,
    log_sinkhorn_converged,
)


def _config():
    from transformers import Mask2FormerConfig
    return Mask2FormerConfig(num_labels=31, train_num_points=32, use_auxiliary_loss=False)


def _weights():
    weights = torch.ones(31)
    weights[1] = 2.0
    weights[19] = 3.0
    return weights


def test_target_mass_is_monotonic_and_sums_to_one():
    loss_fn = AsymmetricPartialOTMask2FormerLoss(
        _config(), _weights(), target_mass_alpha=0.5, target_mass_max=2.0)
    masses = loss_fn.target_column_masses(torch.tensor([0, 1, 19]), queries=100)
    assert torch.allclose(masses[:3], torch.tensor([0.01, 0.015, 0.02]))
    assert masses[0] < masses[1] < masses[2]
    assert torch.allclose(masses.sum(), torch.tensor(1.0))
    assert masses[-1] > 0


def test_zero_alpha_recovers_one_query_mass_per_target():
    loss_fn = AsymmetricPartialOTMask2FormerLoss(
        _config(), _weights(), target_mass_alpha=0.0, target_mass_max=1.0)
    masses = loss_fn.target_column_masses(torch.tensor([0, 1, 19]), queries=100)
    assert torch.allclose(masses[:3], torch.full((3,), 0.01))
    assert torch.allclose(masses[-1], torch.tensor(0.97))


def test_mass_gate_only_boosts_ambiguous_target():
    loss_fn = AsymmetricPartialOTMask2FormerLoss(
        _config(), _weights(), target_mass_alpha=0.5, target_mass_max=2.0)
    masses = loss_fn.target_column_masses(
        torch.tensor([19, 19]), queries=100, mass_gate=torch.tensor([0.0, 1.0]))
    assert torch.allclose(masses[:2], torch.tensor([0.01, 0.02]))


def test_adaptive_gate_prefers_ambiguous_and_small_targets():
    loss_fn = AsymmetricPartialOTMask2FormerLoss(
        _config(), _weights(), mass_mode="adaptive")
    cost = torch.stack([
        torch.tensor([1.0, 0.0, 1.0]),
        torch.tensor([1.0, 8.0, 1.0]),
        torch.tensor([1.0, 8.0, 1.0]),
        torch.tensor([1.0, 8.0, 1.0]),
    ])
    masks = torch.zeros(3, 8, 8)
    masks[0, :2, :2] = 1; masks[1, :4, :4] = 1; masks[2] = 1
    gate, ambiguity, smallness = loss_fn._adaptive_mass_gate(cost, masks)
    assert ambiguity[0] > ambiguity[1]
    assert smallness[0] > smallness[2]
    assert gate[0] > gate[1]


def test_convergence_solver_reaches_strict_marginal_tolerance():
    torch.manual_seed(31)
    cost = 20 * torch.rand(100, 14)
    rows = torch.full((100,), 0.01)
    columns = torch.cat([torch.linspace(0.01, 0.02, 13), torch.tensor([0.805])])
    plan, diagnostics = log_sinkhorn_converged(
        cost, rows, columns, epsilon=0.5, max_iterations=500,
        tolerance=1e-7, check_interval=10, balance_iterations=500)
    assert (plan.sum(1)-rows).abs().max() < 1e-7
    assert (plan.sum(0)-columns).abs().max() < 1e-7
    assert diagnostics["solver_marginal_error"] < 1e-7


def test_dustbin_stays_feasible_at_maximum_semantic_targets():
    loss_fn = AsymmetricPartialOTMask2FormerLoss(
        _config(), torch.full((31,), 3.0), target_mass_alpha=0.5, target_mass_max=2.0)
    masses = loss_fn.target_column_masses(torch.arange(31), queries=100)
    assert torch.allclose(masses[:-1], torch.full((31,), 0.02))
    assert torch.allclose(masses[-1], torch.tensor(0.38), atol=1e-6)


def test_dustbin_fp_penalty_increases_with_risky_foreground_probability():
    loss_fn = AsymmetricPartialOTMask2FormerLoss(
        _config(), _weights(), dustbin_fp_weight=0.1)
    low = torch.zeros(2, 32); low[:, -1] = 8.0
    high = torch.zeros(2, 32); high[:, 19] = 8.0
    dustbin_plan = torch.tensor([0.2, 0.2]); dustbin_mass = torch.tensor(0.4)
    low_loss, low_raw = loss_fn._dustbin_fp_loss(low, dustbin_plan, dustbin_mass)
    high_loss, high_raw = loss_fn._dustbin_fp_loss(high, dustbin_plan, dustbin_mass)
    assert high_raw > low_raw
    assert high_loss > low_loss
    assert torch.allclose(high_loss, 0.1 * high_raw)


def test_zero_fp_weight_removes_only_the_new_fp_term():
    loss_fn = AsymmetricPartialOTMask2FormerLoss(
        _config(), _weights(), dustbin_fp_weight=0.0)
    classes = torch.randn(3, 32)
    fp_loss, raw = loss_fn._dustbin_fp_loss(
        classes, torch.tensor([0.1, 0.2, 0.3]), torch.tensor(0.6))
    assert fp_loss == 0
    assert raw > 0


def test_context_posterior_conditions_target_mass_and_fp_risk():
    prior = torch.zeros(14, 31)
    prior[0, 19] = 1.0
    prior[1, 1] = 1.0
    loss_fn = AsymmetricPartialOTMask2FormerLoss(
        _config(), _weights(), mass_mode="context", station_class_prior=prior,
        target_mass_alpha=0.5, target_mass_max=2.0, context_fp_strength=1.0)
    station_zero = torch.zeros(14); station_zero[0] = 1.0
    relevance = loss_fn._context_relevance(station_zero)
    assert relevance[19] > relevance[1]
    masses = loss_fn.target_column_masses(
        torch.tensor([1, 19]), queries=100, mass_gate=relevance[torch.tensor([1, 19])])
    assert masses[1] > masses[0]
    classes = torch.zeros(2, 32); classes[:, 1] = 8.0
    plan = torch.tensor([0.2, 0.2])
    _, incompatible = loss_fn._dustbin_fp_loss(classes, plan, torch.tensor(0.4), relevance)
    _, neutral = loss_fn._dustbin_fp_loss(classes, plan, torch.tensor(0.4), torch.ones(31))
    assert incompatible > neutral


def test_minmax_context_bounds_are_conservative_on_both_risk_sides():
    prior = torch.zeros(14, 31)
    prior[0, 19] = 1.0
    prior[1, 19] = 0.2
    loss_fn = AsymmetricPartialOTMask2FormerLoss(
        _config(), _weights(), mass_mode="context", station_class_prior=prior,
        context_bound_mode="minmax", target_mass_alpha=0.5,
        target_mass_max=2.0, context_fp_strength=1.0)
    confident = torch.zeros(14); confident[0] = 1.0
    uncertain = torch.zeros(14); uncertain[1] = 1.0
    lower, upper = loss_fn._context_relevance_bounds(
        torch.stack([confident, uncertain]))
    point, _ = loss_fn._context_relevance_bounds(confident)
    assert lower[19] < point[19]
    assert upper[19] == point[19]
    robust_mass = loss_fn.target_column_masses(
        torch.tensor([19]), queries=100, mass_gate=lower[torch.tensor([19])])
    point_mass = loss_fn.target_column_masses(
        torch.tensor([19]), queries=100, mass_gate=point[torch.tensor([19])])
    assert robust_mass[0] < point_mass[0]
    classes = torch.zeros(1, 32); classes[:, 19] = 8.0
    plan = torch.tensor([0.5])
    _, robust_fp = loss_fn._dustbin_fp_loss(classes, plan, torch.tensor(0.5), upper)
    _, aggressive_fp = loss_fn._dustbin_fp_loss(classes, plan, torch.tensor(0.5), lower)
    assert robust_fp < aggressive_fp


def test_pc_apot_backward_and_diagnostics():
    loss_fn = AsymmetricPartialOTMask2FormerLoss(
        _config(), _weights(), epsilon=0.5, iterations=120,
        target_mass_alpha=0.5, target_mass_max=2.0, dustbin_fp_weight=0.1)
    masks = torch.randn(1, 6, 8, 12, requires_grad=True)
    classes = torch.randn(1, 6, 32, requires_grad=True)
    targets = [torch.stack([torch.ones(8, 12), torch.eye(8, 12)])]
    labels = [torch.tensor([1, 19])]
    losses = loss_fn(masks, classes, targets, labels)
    total = sum(losses.values()); total.backward()
    assert set(losses) == {
        "loss_cross_entropy", "loss_mask", "loss_dice", "loss_dustbin_fp"}
    assert torch.isfinite(total)
    assert masks.grad is not None and torch.isfinite(masks.grad).all()
    assert classes.grad is not None and torch.isfinite(classes.grad).all()
    assert loss_fn.last_diagnostics["marginal_error"] < 1e-5
    assert 0 < loss_fn.last_diagnostics["dustbin_mass"] < 1
    assert loss_fn.last_diagnostics["target_mass_max"] > loss_fn.last_diagnostics["target_mass_min"]
    assert set(loss_fn.last_class_diagnostics) == {"1", "19"}
    for values in loss_fn.last_class_diagnostics.values():
        assert abs(values["requested_mass"] - values["transported_mass"]) < 1e-5
