import numpy as np

from scripts.export_safe_graph_oof import (
    gt_component_rows, predicted_component_rows, validation_records_from_reference,
)
from tiger_models.data import LabelMap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def synthetic_case():
    probability = np.full((31, 8, 8), 1e-4, dtype=np.float32)
    target = np.zeros((8, 8), dtype=np.uint8)
    target[1:4, 1:4] = 3
    prediction = np.zeros((8, 8), dtype=np.uint8)
    prediction[1:4, 1:4] = 3
    probability[0] = 0.2
    probability[3, 1:4, 1:4] = 0.9
    probability /= probability.sum(0, keepdims=True)
    entropy = -(probability * np.log(np.maximum(probability, 1e-8))).sum(0)
    return prediction, target, probability, entropy


def test_predicted_component_raw_overlap_is_auditable():
    labelmap = LabelMap.load(ROOT / "data/labelmap.csv")
    prediction, target, probability, entropy = synthetic_case()
    rows = predicted_component_rows(
        prediction, target, probability, entropy, 0, "case", "frame.png",
        np.zeros(14, dtype=np.uint8), labelmap.fine_names,
    )
    assert len(rows) == 1
    assert rows[0]["predicted_class_id"] == 3
    assert rows[0]["same_class_gt_fraction"] == 1.0
    assert rows[0]["error_label"] == "UNADJUDICATED"


def test_gt_component_marks_raw_coverage_not_final_error():
    labelmap = LabelMap.load(ROOT / "data/labelmap.csv")
    prediction, target, _, _ = synthetic_case()
    prediction[:] = 0
    rows = gt_component_rows(prediction, target, 0, "case", "frame.png", labelmap.fine_names)
    assert len(rows) == 1
    assert rows[0]["complete_miss_candidate"] is True
    assert rows[0]["same_class_prediction_coverage"] == 0.0
    assert rows[0]["error_label"] == "UNADJUDICATED"


def test_predicted_component_exports_requested_dense_class_means():
    labelmap = LabelMap.load(ROOT / "data/labelmap.csv")
    prediction, target, probability, entropy = synthetic_case()
    rows = predicted_component_rows(
        prediction, target, probability, entropy, 0, "case", "frame.png",
        np.zeros(14, dtype=np.uint8), labelmap.fine_names, (7, 19),
    )
    assert rows[0]["explicit_probability_class_ids"] == "7|19"
    values = [float(value) for value in rows[0]["explicit_class_mean_probabilities"].split("|")]
    component = prediction == 3
    assert np.allclose(values, [probability[7, component].mean(), probability[19, component].mean()])


def test_reference_records_bypass_unrelated_visibility_rows():
    records = validation_records_from_reference(
        ROOT / "data", 0,
        ROOT / "artifacts/safe_graph_oof/improved_v2_validation_v1",
    )
    assert len(records) == 14
    assert {record.case_id for record in records} == {"center_1_case_11"}
    assert all(record.visibility.shape == (14,) for record in records)
