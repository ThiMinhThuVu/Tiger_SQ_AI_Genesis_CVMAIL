import numpy as np

from scripts.evaluate_safe_graph_aa018_context_cf005 import (
    pair_touch_measurements,
    relation_measurements,
    station_from_frame,
)


ENVELOPE = {"distance_q05": 0.0, "distance_q95": 0.1, "touch_probability": 0.8}


def test_missing_esophagus_is_unknown_not_violation():
    prediction = np.zeros((10, 10), dtype=np.uint8)
    focus = np.zeros_like(prediction, dtype=bool)
    focus[5, 5] = True
    result = relation_measurements(prediction, focus, 7, ENVELOPE)
    assert result["aa018_evidence_available"] == 0
    assert result["aa018_distance_violation"] == 0
    assert result["aa018_touch_mismatch"] == 0


def test_direction_conditioned_aa018_features():
    prediction = np.zeros((10, 10), dtype=np.uint8)
    prediction[5, 5] = 6
    focus = np.zeros_like(prediction, dtype=bool)
    focus[5, 6] = True
    pred7 = relation_measurements(prediction, focus, 7, ENVELOPE)
    pred19 = relation_measurements(prediction, focus, 19, ENVELOPE)
    assert pred7["aa018_touch"] == 1
    assert pred7["aa018_pred7_violation"] == 0
    assert pred19["aa018_pred19_touch"] == 1
    assert pred19["aa018_pred19_compatibility"] == 1


def test_pair_touch_uses_opposite_member_of_cf005():
    prediction = np.zeros((10, 10), dtype=np.uint8)
    prediction[5, 5] = 19
    focus = np.zeros_like(prediction, dtype=bool)
    focus[5, 6] = True
    result = pair_touch_measurements(prediction, focus, 7)
    assert result["pair_partner_available"] == 1
    assert result["pair_partner_touch"] == 1


def test_station_parser_preserves_left_right_suffix():
    assert station_from_frame("center_1_case_11", "center_1_case_11_13L.png") == "13L"
    assert station_from_frame("center_1_case_11", "center_1_case_11_13R.png") == "13R"
