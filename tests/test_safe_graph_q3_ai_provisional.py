import numpy as np

from scripts.evaluate_safe_graph_q3_ai_provisional import (
    confusion_features,
    decision,
    pair_key,
    recover_global_components,
    relation_features_for_frame,
    reliability,
)


def test_pair_key_is_symmetric():
    assert pair_key(19, 7) == (7, 19)


def test_confusion_feature_uses_reviewed_alternative_only():
    row = {
        "predicted_class_id": "7",
        "alternative_class_ids": "7|19|3",
        "alternative_mean_probabilities": "0.6|0.3|0.1",
        "frame": "example.png",
    }
    reviewed = {(7, 19): {}}
    result = confusion_features(row, reviewed, {(7, 19)}, {(7, 19): 0.5})
    assert np.isclose(result["confusion_all_score"], 0.15)
    assert np.isclose(result["confusion_valid_score"], 0.15)


def test_missing_relation_partner_is_unknown_not_violation():
    prediction = np.zeros((10, 10), dtype=np.uint8)
    prediction[2:5, 2:5] = 3
    frame_rows = [{"component_id": "1", "predicted_class_id": "3"}]
    result = relation_features_for_frame(
        prediction, frame_rows,
        {(3, 14): {"distance_q05": 0.0, "distance_q95": 0.1,
                   "touch_probability": 1.0, "observations": 10.0}},
        {(3, 14)},
    )[1]
    assert result["relation_evidence_available"] == 0.0
    assert result["relation_distance_violation"] == 0.0
    assert result["relation_touch_mismatch"] == 0.0


def test_recover_global_components_follows_class_order():
    prediction = np.zeros((10, 10), dtype=np.uint8)
    prediction[1:3, 1:3] = 2
    prediction[6:8, 1:3] = 2
    prediction[4:7, 6:9] = 7
    components = recover_global_components(prediction)
    assert len(components) == 3
    assert components[3].sum() == 9


def test_reliability_decreases_with_entropy_and_tool_probability():
    good = {"mean_entropy": "0.1", "mean_tool_probability": "0.0"}
    poor = {"mean_entropy": "2.5", "mean_tool_probability": "0.8"}
    assert reliability(good) > reliability(poor)


def test_decision_requires_gain_over_class_identity_control():
    metric_rows = []
    macro = {
        "Q0_confidence": 0.25,
        "Q3e_local_plus_gated_knowledge": 0.48,
        "QCTRL_local_plus_class_identity": 0.72,
        "Q3f_local_class_plus_gated_knowledge": 0.71,
    }
    for method, auprc in macro.items():
        metric_rows.append({
            "target": "dangerous_error", "method": method,
            "scope": "macro_case_mean", "case_id": "ALL", "auprc": auprc,
        })
    for case_index in range(5):
        for method, auprc in macro.items():
            metric_rows.append({
                "target": "dangerous_error", "method": method,
                "scope": "per_case", "case_id": f"case_{case_index}", "auprc": auprc,
            })
    result = decision(metric_rows)
    assert result["status"] == "NO_GO_CURRENT_Q3_FEATURES"
    assert result["knowledge_absolute_delta_over_class_control"] < 0
