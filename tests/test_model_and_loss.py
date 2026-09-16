from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from tiger_models.losses import PrimarySAMLoss
from tiger_models.multitask_sam import MultitaskSAM
from tiger_models.runtime import load_checkpoint, save_checkpoint
from tiger_models.sam_semantic_decoder import SAMSemanticDecoder


class FakeEncoder(nn.Module):
    feature_channels = (8, 8, 8, 8)

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, 8, 1)

    def forward(self, image: torch.Tensor) -> dict[str, object]:
        base = self.projection(image)
        features = [F.avg_pool2d(base, 2**level) for level in (2, 3, 4, 5)]
        return {"features": features, "padding": (0, 0), "padded_size": image.shape[-2:]}


def build_fake_model(coarse_head: bool = False) -> MultitaskSAM:
    return MultitaskSAM(encoder=FakeEncoder(), decoder_channels=16, coarse_head=coarse_head)


def test_primary_full_resolution_shapes() -> None:
    model = build_fake_model().eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 512, 896))
    assert output["fine_logits"].shape == (1, 31, 512, 896)
    assert output["visibility_logits"].shape == (1, 14)
    assert "coarse_logits" not in output


def test_optional_coarse_ablation_shape() -> None:
    model = build_fake_model(coarse_head=True).eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 64, 96))
    assert output["coarse_logits"].shape == (1, 16, 64, 96)


def test_loss_backward_and_finite_probabilities() -> None:
    torch.manual_seed(2026)
    model = build_fake_model()
    image = torch.randn(1, 3, 64, 96)
    fine = torch.randint(0, 31, (1, 64, 96))
    visibility = torch.randint(0, 2, (1, 14)).float()
    criterion = PrimarySAMLoss(torch.ones(31), torch.ones(14))
    output = model(image)
    losses = criterion(output, fine, visibility)
    assert {"boundary_bce", "boundary_dice", "boundary_loss"} <= set(losses)
    assert torch.allclose(losses["boundary_loss"], losses["boundary_bce"] + losses["boundary_dice"])
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_checkpoint_roundtrip_and_deterministic_inference(tmp_path: Path) -> None:
    torch.manual_seed(2026)
    model = build_fake_model().eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    sample = torch.randn(1, 3, 64, 96)
    with torch.inference_mode():
        expected = model(sample)["fine_logits"]
        repeated = model(sample)["fine_logits"]
    assert torch.equal(expected, repeated)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, scheduler, scaler, 0, {"test": True}, {}, [])
    reloaded = build_fake_model().eval()
    load_checkpoint(path, reloaded, device="cpu")
    with torch.inference_mode():
        actual = reloaded(sample)["fine_logits"]
    assert torch.equal(expected, actual)
