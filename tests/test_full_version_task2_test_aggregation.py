from full_version.task2_fine_31cls.scripts.evaluate_test import _case_values, _score


def test_case_values_reject_duplicate_oof_case():
    folds = [
        {"fine": {"per_case_dice": {"center_1_case_2": 0.5}}},
        {"fine": {"per_case_dice": {"center_1_case_2": 0.6}}},
    ]
    try:
        _case_values(folds, "fine", "dice")
    except ValueError as error:
        assert "more than once" in str(error)
    else:
        raise AssertionError("duplicate OOF case was not rejected")


def test_task_score_definition():
    assert _score(0.8, 0.2) == 0.8
