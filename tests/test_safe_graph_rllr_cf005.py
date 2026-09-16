import numpy as np

from scripts.evaluate_safe_graph_rllr_cf005 import (
    score_relations,
    shuffled_partner_map,
    station_mixture_log_compatibility,
    subgroup_metrics,
)


def relation_row(low, high, touch, observations=20):
    return {
        "boundary_distance_norm_q05": str(low),
        "boundary_distance_norm_q95": str(high),
        "touch_probability_when_covisible": str(touch),
        "relation_observations": str(observations),
    }


def test_compatibility_prefers_matching_distance_and_touch():
    rows = [relation_row(0.0, 0.02, 0.9)]
    near_touch = station_mixture_log_compatibility(0.01, 1.0, rows, 100.0)
    far_no_touch = station_mixture_log_compatibility(0.5, 0.0, rows, 100.0)
    assert near_touch > far_no_touch


def test_rllr_is_oriented_toward_counterpart_hypothesis():
    prediction = np.zeros((20, 20), dtype=np.uint8)
    prediction[10, 10] = 6
    focus = np.zeros_like(prediction, dtype=bool)
    focus[10, 11] = True
    profile = {
        "eligible": [6],
        "rows": {
            7: {6: [relation_row(0.0, 0.02, 0.9)]},
            19: {6: [relation_row(0.3, 0.5, 0.1)]},
        },
    }
    result = score_relations(prediction, focus, 19, profile, 20.0)
    assert result["rllr"] > 0  # Observed relation supports counterpart H7.
    assert result["rllr_visible_partners"] == 1


def test_missing_partners_are_unknown():
    prediction = np.zeros((20, 20), dtype=np.uint8)
    focus = np.zeros_like(prediction, dtype=bool)
    focus[10, 11] = True
    profile = {"eligible": [6], "rows": {7: {6: []}, 19: {6: []}}}
    result = score_relations(prediction, focus, 19, profile, 20.0)
    assert result == {"rllr": 0.0, "rllr_visible_partners": 0.0, "rllr_evidence_available": 0.0}


def test_shuffled_map_is_deterministic_and_preserves_labels():
    first = shuffled_partner_map([3, 4, 5, 6], 0, 2026)
    second = shuffled_partner_map([3, 4, 5, 6], 0, 2026)
    assert first == second
    assert set(first) == set(first.values()) == {3, 4, 5, 6}


def test_subgroup_metrics_skips_empty_unknown_group():
    rows = [{
        "predicted_class_id": "7", "area_bin": "tiny",
        "rllr_evidence_available": 1.0, "rllr_reliability": 0.6,
    }]
    output = subgroup_metrics(
        rows, np.asarray([1]), {"m": np.asarray([0.5])}, np.asarray(["case"]),
    )
    assert not any(row["group"] == "rllr_unknown" for row in output)
