from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tiger_models.multitask_sam import MultitaskSAM


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.integration
def test_real_sam2_checkpoint_interface_and_freeze_policy() -> None:
    model = MultitaskSAM(
        checkpoint_path=str(ROOT / "checkpoints/sam2.1_hiera_tiny.pt"),
        finetune="partial", gradient_checkpointing=True, decoder_channels=32,
    ).eval()
    policy = model.encoder.freeze_policy()
    assert policy["trainable_trunk_blocks"] == [10, 11]
    assert policy["trainable_neck"] is True
    with torch.inference_mode():
        first = model(torch.zeros(1, 3, 64, 96))
        second = model(torch.zeros(1, 3, 64, 96))
    assert first["fine_logits"].shape == (1, 31, 64, 96)
    assert first["visibility_logits"].shape == (1, 14)
    assert torch.equal(first["fine_logits"], second["fine_logits"])
