import numpy as np

from scripts.analyze_safe_graph_p2_cf005 import (
    confusion_scaling_diagnostic,
    enrich_rows,
    summarize_operational,
)


def test_enrich_rows_derives_pair_share_and_direction():
    rows = [{
        "local_confidence_risk": "0.6",
        "counterpart_probability": "0.1",
        "predicted_class_id": "7",
    }]
    result = enrich_rows(rows)[0]
    assert np.isclose(result["current_class_probability"], 0.4)
    assert np.isclose(result["pair_probability_share"], 0.2)
    assert result["direction"] == "pred_fatty_esophagus_gt_lymph_node"


def test_confusion_scaling_identifies_fold_constant_multiplier():
    rows = [
        {"fold": "0", "counterpart_probability": "0.2", "fold_excluded_confusion_score": "0.1"},
        {"fold": "0", "counterpart_probability": "0.4", "fold_excluded_confusion_score": "0.2"},
    ]
    result = confusion_scaling_diagnostic(rows)[0]
    assert np.isclose(result["confusion_to_counterpart_ratio_min"], 0.5)
    assert np.isclose(result["confusion_to_counterpart_ratio_max"], 0.5)
    assert np.isclose(result["pearson_correlation"], 1.0)


def test_operational_summary_preserves_coverage_level():
    rows = [
        {"method": "P2", "target_coverage": 0.1, "warning_coverage": 0.1,
         "warning_ppv": 0.5, "warning_recall": 0.2, "warning_count": 2},
        {"method": "P2", "target_coverage": 0.1, "warning_coverage": 0.2,
         "warning_ppv": 0.0, "warning_recall": 0.0, "warning_count": 1},
    ]
    result = summarize_operational(rows)[0]
    assert result["target_coverage"] == 0.1
    assert np.isclose(result["mean_actual_coverage"], 0.15)
    assert result["cases_zero_true_positive_warning"] == 1
