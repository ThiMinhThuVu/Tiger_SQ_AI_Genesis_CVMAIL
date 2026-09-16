from __future__ import annotations

import pytest
import torch

from tiger_models.cetl_loss import cross_entropy_tversky_loss


def test_cetl_is_finite_and_differentiable_with_resized_target():
    logits = torch.randn(2, 4, 5, 7, requires_grad=True)
    probabilities = logits.softmax(1)
    target = torch.tensor(
        [
            [[0, 0, 1, 1], [0, 2, 2, 1], [0, 0, 2, 1]],
            [[0, 3, 3, 3], [0, 0, 3, 3], [0, 0, 0, 3]],
        ]
    )
    loss, terms = cross_entropy_tversky_loss(
        probabilities,
        target,
        class_weights=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        ce_fraction=0.5,
        alpha=0.3,
        beta=0.7,
    )
    loss.backward()

    assert set(terms) == {"cross_entropy", "tversky"}
    assert torch.isfinite(loss)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_absent_class_is_excluded_only_from_tversky_average():
    target = torch.zeros(1, 2, 2, dtype=torch.long)
    good = torch.tensor([[[[0.9, 0.9], [0.9, 0.9]], [[0.1, 0.1], [0.1, 0.1]]]])
    bad = torch.tensor([[[[0.6, 0.6], [0.6, 0.6]], [[0.4, 0.4], [0.4, 0.4]]]])

    good_loss, good_terms = cross_entropy_tversky_loss(good, target)
    bad_loss, bad_terms = cross_entropy_tversky_loss(bad, target)

    assert bad_loss > good_loss
    assert bad_terms["cross_entropy"] > good_terms["cross_entropy"]
    assert bad_terms["tversky"] > good_terms["tversky"]


def test_cetl_rejects_invalid_mix_fraction():
    probabilities = torch.full((1, 2, 2, 2), 0.5)
    target = torch.zeros(1, 2, 2, dtype=torch.long)
    with pytest.raises(ValueError, match="ce_fraction"):
        cross_entropy_tversky_loss(probabilities, target, ce_fraction=1.1)
