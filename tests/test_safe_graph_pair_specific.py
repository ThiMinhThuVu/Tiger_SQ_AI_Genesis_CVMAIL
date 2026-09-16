from scripts.evaluate_safe_graph_pair_specific import (
    counterpart_probability,
    feasibility_for_pair,
    pair_target,
    operational_summary,
    threshold_metrics,
)


def test_pair_target_requires_exact_bidirectional_swap():
    row = {
        "predicted_class_id": "7", "dominant_gt_class_id": "19",
        "error_label": "CLASS_SWAP",
    }
    assert pair_target(row, 7, 19)
    row["dominant_gt_class_id"] = "3"
    assert not pair_target(row, 7, 19)


def test_counterpart_probability_returns_zero_if_not_topk():
    row = {
        "alternative_class_ids": "7|3|4",
        "alternative_mean_probabilities": "0.8|0.1|0.05",
    }
    assert counterpart_probability(row, 19) == 0.0


def test_feasibility_gate_counts_case_support_and_nontiny():
    rows = []
    for case_index in range(3):
        rows.append({
            "case_id": f"case_{case_index}", "predicted_class_id": "7",
            "dominant_gt_class_id": "19", "error_label": "CLASS_SWAP",
            "area_fraction": "0.001",
        })
        rows.append({
            "case_id": f"case_{case_index}", "predicted_class_id": "7",
            "dominant_gt_class_id": "7", "error_label": "CORRECT_COMPONENT",
            "area_fraction": "0.001",
        })
    pair = {
        "confusion_rule_id": "CF-X", "class_i": "7", "name_i": "A",
        "class_j": "19", "name_j": "B",
        "expert_decision_valid_invalid_depends": "VALID",
        "expert_warning_eligible": "YES",
    }
    summary, _ = feasibility_for_pair(rows, pair, {(7, 19)}, 3, 3, 3, 0.25)
    assert summary["eligible"] is True


def test_threshold_metrics_reports_ppv_recall_and_coverage():
    import numpy as np
    target = np.asarray([1, 0, 1, 0], dtype=np.uint8)
    risk = np.asarray([0.9, 0.8, 0.2, 0.1])
    result = threshold_metrics(target, risk, 0.5)
    assert result["warning_coverage"] == 0.5
    assert result["warning_ppv"] == 0.5
    assert result["warning_recall"] == 0.5


def test_operational_summary_counts_zero_ppv_cases():
    rows = [
        {"method": "P5", "warning_coverage": 0.1, "warning_ppv": 0.0,
         "warning_recall": 0.0, "warning_count": 2},
        {"method": "P5", "warning_coverage": 0.1, "warning_ppv": 0.5,
         "warning_recall": 0.2, "warning_count": 2},
    ]
    result = operational_summary(rows)[0]
    assert result["mean_warning_ppv"] == 0.25
    assert result["cases_zero_true_positive_warning"] == 1
