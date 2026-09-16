from __future__ import annotations

import torch

from full_version.task2_fine_31cls.scripts.p3_cetl_gradient_audit import (
    gradient_comparison,
    select_records,
    summarize_batches,
)


def test_select_records_is_deterministic_and_without_replacement():
    records = list(range(20))
    first = select_records(records, 8, 17)
    second = select_records(records, 8, 17)
    assert first == second
    assert len(set(first)) == 8


def test_gradient_comparison_reports_norm_ratio_and_cosine():
    native = (torch.tensor([3.0, 4.0]),)
    cetl = (torch.tensor([6.0, 8.0]),)
    result = gradient_comparison(native, cetl, [0], cetl_weight=0.25)
    assert result["native_gradient_norm"] == 5.0
    assert result["cetl_gradient_norm"] == 10.0
    assert result["raw_norm_ratio"] == 2.0
    assert result["effective_norm_ratio"] == 0.5
    assert abs(result["cosine_similarity"] - 1.0) < 1e-7


def test_summary_gate_rescales_only_when_underweighted_and_aligned():
    rows = []
    for batch in range(2):
        rows.append({
            "gradient_groups": {
                name: {
                    "native_gradient_norm": 10.0,
                    "cetl_gradient_norm": 1.0,
                    "raw_norm_ratio": 0.1,
                    "effective_norm_ratio": 0.025,
                    "cosine_similarity": 0.2,
                }
                for name in ("class_predictor", "transformer_decoder", "backbone_stage4")
            }
        })
    summary = summarize_batches(rows, cetl_weight=0.25)
    gate = summary["mechanical_gate"]
    assert gate["decision"] == "RUN_ONE_GRADIENT_CALIBRATED_CETL_VARIANT"
    assert abs(gate["recommended_cetl_weight_if_rescaled"] - 0.75) < 1e-7
