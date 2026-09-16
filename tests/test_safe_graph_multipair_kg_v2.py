import numpy as np

from scripts.evaluate_safe_graph_multipair_kg_v2 import (
    aggregate_subset_risk,
    candidate_rows,
    deterministic_partner_map,
    eligible_partners,
    hypothesis_relation_score,
)


def relation_row(observations=20, low=0.0, high=0.02, touch=0.9):
    return {
        "relation_observations": str(observations),
        "boundary_distance_norm_q05": str(low),
        "boundary_distance_norm_q95": str(high),
        "touch_probability_when_covisible": str(touch),
    }


def test_candidate_target_requires_exact_dominant_alternative():
    row = {
        "fold": "0", "case_id": "case", "frame": "case_1.png", "component_id": "1",
        "clinical_risk_weight": "3", "predicted_class_id": "19",
        "alternative_class_ids": "19|7|6", "alternative_mean_probabilities": "0.6|0.3|0.1",
        "error_label": "CLASS_SWAP", "dominant_gt_class_id": "7", "dominant_gt_fraction": "0.8",
    }
    output = candidate_rows([row], 3.0)
    assert len(output) == 2
    assert [item["exact_pair_swap_target"] for item in output] == [True, False]


def test_candidate_rows_excludes_background_from_anatomy_kg():
    row = {
        "fold": "0", "case_id": "case", "frame": "case_1.png", "component_id": "1",
        "clinical_risk_weight": "3", "predicted_class_id": "19",
        "alternative_class_ids": "19|0|7", "alternative_mean_probabilities": "0.6|0.25|0.15",
        "error_label": "CLASS_SWAP", "dominant_gt_class_id": "7", "dominant_gt_fraction": "0.8",
    }
    output = candidate_rows([row], 3.0)
    assert len(output) == 1
    assert output[0]["alternative_class_id"] == 7
    assert output[0]["exact_pair_swap_target"] is True


def test_eligible_partners_requires_support_for_both_hypotheses():
    profile = {class_id: {} for class_id in range(1, 31)}
    profile[7] = {6: [relation_row(20)] * 3, 10: [relation_row(20)] * 3}
    profile[19] = {6: [relation_row(20)] * 3, 10: [relation_row(1)]}
    assert eligible_partners(profile, 7, 19, 20, 3) == [6]


def test_missing_partner_is_unknown_zero_evidence():
    prediction = np.zeros((20, 20), dtype=np.uint8)
    focus = np.zeros_like(prediction, dtype=bool); focus[10, 10] = True
    profile = {class_id: {} for class_id in range(1, 31)}
    profile[7] = {6: [relation_row()]}; profile[19] = {6: [relation_row()]}
    result = hypothesis_relation_score(
        prediction, focus, 19, 7, profile, {}, [6], 20.0,
    )
    assert result["kg_llr"] == 0
    assert result["kg_evidence_available"] == 0


def test_contrast_orients_toward_alternative_hypothesis():
    prediction = np.zeros((20, 20), dtype=np.uint8); prediction[10, 10] = 6
    focus = np.zeros_like(prediction, dtype=bool); focus[10, 11] = True
    profile = {class_id: {} for class_id in range(1, 31)}
    profile[19] = {6: [relation_row(low=0.3, high=0.5, touch=0.1)]}
    profile[7] = {6: [relation_row(low=0.0, high=0.02, touch=0.9)]}
    distance_maps = {6: np.sqrt((np.indices(prediction.shape)[0] - 10) ** 2 + (np.indices(prediction.shape)[1] - 10) ** 2)}
    result = hypothesis_relation_score(
        prediction, focus, 19, 7, profile, distance_maps, [6], 20.0,
    )
    assert result["kg_llr"] > 0


def test_shuffled_partner_map_is_deterministic_per_pair():
    first = deterministic_partner_map([3, 4, 5, 6], 0, 19, 7, 2026)
    second = deterministic_partner_map([3, 4, 5, 6], 0, 19, 7, 2026)
    assert first == second
    assert set(first) == set(first.values()) == {3, 4, 5, 6}


def test_aggregate_subset_risk_maps_subset_values_to_global_rows():
    rows = [
        {"fold": 0, "case_id": "a", "frame": "f", "component_id": 1},
        {"fold": 0, "case_id": "b", "frame": "g", "component_id": 2},
        {"fold": 0, "case_id": "a", "frame": "f", "component_id": 1},
        {"fold": 0, "case_id": "b", "frame": "h", "component_id": 3},
    ]
    selected = np.asarray([True, False, True, False])
    result = aggregate_subset_risk(rows, np.asarray([0.2, 0.8]), selected)
    np.testing.assert_allclose(result, [0.8])
