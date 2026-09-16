import numpy as np

from scripts.evaluate_safe_graph_p2_dense_cf005 import (
    explicit_probability_lookup,
    merge_dense_cohort,
    recovery_summary,
)


def test_explicit_probability_lookup_requires_pair_classes():
    row = {
        "explicit_probability_class_ids": "7|19",
        "explicit_class_mean_probabilities": "0.2|0.3",
    }
    assert explicit_probability_lookup(row) == {7: 0.2, 19: 0.3}


def test_merge_dense_cohort_selects_counterpart_by_prediction():
    pair = [{
        "fold": "0", "case_id": "case", "frame": "frame.png", "component_id": "1",
        "predicted_class_id": "7", "counterpart_probability": "0",
        "pair_swap_target": "True", "area_bin": "small",
    }]
    dense = [{
        "fold": "0", "case_id": "case", "frame": "frame.png", "component_id": "1",
        "predicted_class_id": "7", "explicit_probability_class_ids": "7|19",
        "explicit_class_mean_probabilities": "0.6|0.1",
    }]
    result = merge_dense_cohort(pair, dense)[0]
    assert np.isclose(result["dense_current_probability"], 0.6)
    assert np.isclose(result["dense_counterpart_probability"], 0.1)
    assert np.isclose(result["dense_pair_probability_share"], 1 / 7)


def test_recovery_summary_counts_topk_missing_positive():
    rows = [{
        "predicted_class_id": "7", "pair_swap_target": "True", "area_bin": "small",
        "topk_counterpart_available": False, "dense_counterpart_probability": 0.01,
    }]
    result = recovery_summary(rows)[0]
    assert result["topk_missing_positives"] == 1
    assert result["dense_nonzero_positives"] == 1
