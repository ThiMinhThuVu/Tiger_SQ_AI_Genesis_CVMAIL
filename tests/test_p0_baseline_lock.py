import math

from full_version.task2_fine_31cls.scripts.p0_baseline_lock import (
    aggregate_rows,
    synthetic_metric_parity,
    task_score,
)


def _frame(name, case_id, center, fold, dice, nhd):
    return {
        "name": name,
        "case_id": case_id,
        "center": center,
        "fold": fold,
        "width": 8,
        "height": 6,
        "dice": dice,
        "nhd": nhd,
        "task_score": task_score(dice, nhd),
    }


def _class(name, case_id, center, fold, class_id, weight, dice, nhd):
    return {
        "name": name,
        "case_id": case_id,
        "center": center,
        "fold": fold,
        "class_id": class_id,
        "class_name": f"class_{class_id}",
        "weight": weight,
        "target_present": 1,
        "prediction_present": 1,
        "target_pixels": 4,
        "prediction_pixels": 4,
        "dice": dice,
        "nhd": nhd,
    }


def test_synthetic_fast_metrics_match_official_reference():
    result = synthetic_metric_parity()
    assert result["status"] == "PASS"
    assert result["max_absolute_dice_difference"] <= result["tolerance"]
    assert result["max_absolute_nhd_difference"] <= result["tolerance"]


def test_aggregate_is_case_balanced_not_frame_balanced():
    frame_rows = [
        _frame("a0.png", "case_a", 1, 0, 1.0, 0.0),
        _frame("a1.png", "case_a", 1, 0, 1.0, 0.0),
        _frame("b0.png", "case_b", 2, 1, 0.0, 1.0),
    ]
    # Sixty unit-weight pseudo-classes make the class-error reconstruction exact.
    class_rows = []
    for frame in frame_rows:
        for class_id in range(60):
            class_rows.append(_class(
                frame["name"], frame["case_id"], frame["center"], frame["fold"],
                class_id, 1.0, frame["dice"], frame["nhd"],
            ))
    result = aggregate_rows(frame_rows, class_rows)
    assert result["overall"]["dice"] == 0.5
    assert result["overall"]["nhd"] == 0.5
    assert result["overall"]["task_score"] == 0.5
    assert result["overall"]["task_score"] != 2.0 / 3.0
    assert result["aggregation_absolute_difference"] == 0.0


def test_task_score_definition():
    assert math.isclose(task_score(0.734709, 0.186656), 0.7740265)
