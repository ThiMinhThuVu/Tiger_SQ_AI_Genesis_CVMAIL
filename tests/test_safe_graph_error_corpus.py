import numpy as np

from scripts.build_safe_graph_error_corpus import label_frame
from tiger_models.data import LabelMap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LABELMAP = LabelMap.load(ROOT / "data/labelmap.csv")
WEIGHTS = LABELMAP.fine_weights.astype(float)


def run(prediction, target):
    return label_frame(prediction, target, list(LABELMAP.fine_names), WEIGHTS, 0.5, 0.5)


def test_exact_component_is_correct_and_detected():
    target = np.zeros((12, 12), dtype=np.uint8)
    target[2:8, 2:8] = 3
    predicted, gt = run(target.copy(), target)
    assert predicted[0]["error_label"] == "CORRECT_COMPONENT"
    assert gt[0]["error_label"] == "DETECTED_COMPONENT"


def test_absent_class_prediction_is_hallucination():
    target = np.zeros((12, 12), dtype=np.uint8)
    prediction = np.zeros_like(target)
    prediction[2:8, 2:8] = 13
    predicted, _ = run(prediction, target)
    assert predicted[0]["error_label"] == "ABSENT_CLASS_HALLUCINATION"
    assert predicted[0]["dangerous_error_candidate"] is True


def test_wrong_class_overlap_is_class_swap():
    target = np.zeros((12, 12), dtype=np.uint8)
    target[2:8, 2:8] = 3
    prediction = np.zeros_like(target)
    prediction[2:8, 2:8] = 4
    predicted, gt = run(prediction, target)
    assert predicted[0]["error_label"] == "CLASS_SWAP"
    assert gt[0]["error_label"] == "COMPLETE_MISS"


def test_split_prediction_creates_fragment():
    target = np.zeros((20, 20), dtype=np.uint8)
    target[2:18, 2:18] = 3
    prediction = np.zeros_like(target)
    prediction[2:18, 2:9] = 3
    prediction[2:18, 11:18] = 3
    predicted, gt = run(prediction, target)
    assert {row["error_label"] for row in predicted} == {
        "CORRECT_COMPONENT", "FRAGMENTED_PREDICTION"
    }
    assert gt[0]["error_label"] == "DETECTED_COMPONENT"
