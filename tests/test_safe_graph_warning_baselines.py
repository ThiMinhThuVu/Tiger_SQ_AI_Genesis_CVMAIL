import numpy as np

from scripts.train_safe_graph_warning_baselines import binary_metrics, features, parse_bool


def test_feature_direction_and_shape():
    row = {
        "area_fraction": "0.01",
        "mean_predicted_class_probability": "0.8",
        "mean_top1_probability": "0.9",
        "mean_top1_top2_margin": "0.7",
        "mean_entropy": "0.4",
    }
    values = features(row)
    assert len(values) == 5
    assert np.allclose(values[:4], [0.2, 0.1, 0.3, 0.4])


def test_binary_metrics_perfect_ranking():
    result = binary_metrics(np.asarray([0, 0, 1, 1]), np.asarray([0.1, 0.2, 0.8, 0.9]))
    assert result["auprc"] == 1.0
    assert result["auroc"] == 1.0


def test_parse_bool_is_explicit():
    assert parse_bool("True") is True
    assert parse_bool("False") is False
