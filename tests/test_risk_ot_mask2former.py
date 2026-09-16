from __future__ import annotations

import torch

from tiger_models.risk_ot_mask2former import log_sinkhorn, RiskSinkhornMask2FormerLoss


def test_sinkhorn_matches_marginals():
    torch.manual_seed(7)
    cost=torch.rand(5,3); rows=torch.full((5,),0.2); columns=torch.tensor([0.2,0.2,0.6])
    plan=log_sinkhorn(cost,rows,columns,epsilon=0.1,iterations=100)
    assert torch.allclose(plan.sum(1),rows,atol=1e-5)
    assert torch.allclose(plan.sum(0),columns,atol=1e-5)


def test_sinkhorn_sharp_cost_is_still_balanced():
    torch.manual_seed(11)
    cost=20*torch.rand(100,14); rows=torch.full((100,),0.01)
    columns=torch.cat([torch.full((13,),0.01),torch.tensor([0.87])])
    plan=log_sinkhorn(cost,rows,columns,epsilon=0.5,iterations=120)
    assert (plan.sum(1)-rows).abs().max() < 1e-5
    assert (plan.sum(0)-columns).abs().max() < 1e-5


def test_risk_ot_loss_backward_and_diagnostics():
    from transformers import Mask2FormerConfig
    config=Mask2FormerConfig(num_labels=31,train_num_points=32,use_auxiliary_loss=False)
    loss_fn=RiskSinkhornMask2FormerLoss(config,torch.tensor([1.]+[2.]*18+[3.]+[1.]*11),
                                       epsilon=0.5,iterations=120)
    masks=torch.randn(1,6,8,12,requires_grad=True); classes=torch.randn(1,6,32,requires_grad=True)
    targets=[torch.stack([torch.ones(8,12),torch.eye(8,12)])]; labels=[torch.tensor([1,19])]
    losses=loss_fn(masks,classes,targets,labels); total=sum(losses.values()); total.backward()
    assert set(losses)=={"loss_cross_entropy","loss_mask","loss_dice"}
    assert torch.isfinite(total) and masks.grad is not None and classes.grad is not None
    assert loss_fn.last_diagnostics["marginal_error"] < 1e-4
    assert 0 <= loss_fn.last_diagnostics["dustbin_mass"] <= 1
