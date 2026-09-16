from __future__ import annotations

import numpy as np
import torch

from tiger_models.improved_mask2former import (aggregate_fine_to_coarse, presence_targets,
    ClinicalMask2FormerLoss, soft_presence_modulation, symmetric_kl)
from tiger_models.metrics import segmentation_failure_metrics


def test_fine_to_coarse_aggregation_preserves_probability_mass():
    fine = torch.rand(2, 31, 8, 12); fine = fine / fine.sum(1, keepdim=True)
    mapping = torch.tensor([0] + [1 + (index % 15) for index in range(30)])
    coarse = aggregate_fine_to_coarse(fine, mapping)
    assert coarse.shape == (2, 16, 8, 12)
    assert torch.allclose(coarse.sum(1), torch.ones_like(coarse[:, 0]), atol=1e-6)


def test_hierarchy_and_presence_are_finite_and_differentiable():
    first = torch.softmax(torch.randn(1, 16, 4, 6, requires_grad=True), 1)
    second = torch.softmax(torch.randn(1, 16, 4, 6), 1)
    loss = symmetric_kl(first, second); loss.backward()
    assert torch.isfinite(loss)
    target = torch.tensor([[[0, 1], [1, 3]], [[0, 0], [2, 2]]])
    presence = presence_targets(target)
    assert presence.shape == (2, 31) and presence[0, 3] == 1 and presence[1, 3] == 0


def test_soft_modulation_never_hard_deletes_a_class():
    probability = torch.full((1, 31, 2, 2), 1 / 31)
    logits = torch.full((1, 31), -100.0)
    result = soft_presence_modulation(probability, logits, epsilon=0.25, gamma=1.0)
    assert torch.all(result > 0)
    assert torch.allclose(result.sum(1), torch.ones_like(result[:, 0]))


def test_failure_metrics_absent_fp_and_complete_miss():
    target = np.array([[0, 1], [0, 0]]); prediction = np.array([[0, 0], [2, 0]])
    metrics = segmentation_failure_metrics([prediction], [target], ["bg", "one", "two"])["per_class"]
    assert metrics[1]["complete_miss_rate"] == 1.0
    assert metrics[2]["absent_fp_rate"] == 1.0


def test_clinical_set_loss_backward():
    from transformers import Mask2FormerConfig
    config = Mask2FormerConfig(num_labels=31, train_num_points=32, use_auxiliary_loss=False)
    criterion = ClinicalMask2FormerLoss(config, torch.arange(1, 32).float())
    masks = torch.randn(1, 5, 8, 12, requires_grad=True)
    classes = torch.randn(1, 5, 32, requires_grad=True)
    target_masks = [torch.stack([torch.ones(8, 12), torch.eye(8, 12)])]
    target_classes = [torch.tensor([1, 19])]
    losses = criterion(masks, classes, target_masks, target_classes)
    total = sum(losses.values()); total.backward()
    assert set(losses) == {"loss_cross_entropy", "loss_mask", "loss_dice"}
    assert torch.isfinite(total) and masks.grad is not None and classes.grad is not None
