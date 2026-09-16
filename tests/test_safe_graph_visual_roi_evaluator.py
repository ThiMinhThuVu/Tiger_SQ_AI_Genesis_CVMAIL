import numpy as np

from scripts.evaluate_safe_graph_visual_roi_cf005 import (
    design_matrices,
    make_shuffled_roi,
    numeric_value,
)


def test_numeric_value_accepts_serialized_booleans():
    assert numeric_value("True") == 1.0
    assert numeric_value("False") == 0.0


def test_shuffled_roi_is_deterministic_and_stays_within_group():
    rows = [
        {"case_id": "a", "predicted_class_id": "7"},
        {"case_id": "a", "predicted_class_id": "7"},
        {"case_id": "a", "predicted_class_id": "19"},
        {"case_id": "b", "predicted_class_id": "7"},
    ]
    roi = np.arange(8, dtype=np.float32).reshape(4, 2)
    first, first_permutation = make_shuffled_roi(rows, roi, 2026)
    second, second_permutation = make_shuffled_roi(rows, roi, 2026)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first_permutation, second_permutation)
    assert set(first_permutation[:2]) == {0, 1}
    assert first_permutation[2] == 2
    assert first_permutation[3] == 3


def test_design_matrix_fits_transform_on_training_rows_only():
    rows = [{"p2": value} for value in [0.0, 1.0, 1000.0]]
    embeddings = {"roi": np.asarray([[0.0, 0.0], [1.0, 1.0], [1000.0, 1000.0]])}
    train = np.asarray([True, True, False])
    test = ~train
    x_train, x_test, audit = design_matrices(
        rows, embeddings, {"scalar": ["p2"], "embedding": ["roi"]},
        train, test, pca_components=1,
    )
    np.testing.assert_allclose(x_train[:, 0], [-1.0, 1.0])
    assert x_test[0, 0] > 1000
    assert x_train.shape == (2, 2)
    assert audit["features_after_transform"] == 2
