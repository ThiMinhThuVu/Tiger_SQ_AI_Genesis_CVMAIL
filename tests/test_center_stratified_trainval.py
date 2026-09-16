from collections import Counter

from full_version.task2_fine_31cls.scripts.data_full import build_trainval_manifest


def test_center_stratified_manifest_is_32_by_8_without_test_split():
    manifest = build_trainval_manifest("data", 2026)
    assert manifest["version"] == "full40_center_stratified_5fold_trainval_d517_v2"
    assert manifest["snapshot"]["frame_count"] == 517
    validation_cases = []
    for fold in manifest["folds"].values():
        train = set(fold["train_cases"])
        validation = set(fold["validation_cases"])
        assert len(train) == 32
        assert len(validation) == 8
        assert train.isdisjoint(validation)
        assert len(train | validation) == 40
        assert "test_cases" not in fold
        centers = set(fold["summary"]["validation"]["centers"])
        assert {1, 2, 3, 7}.issubset(centers)
        assert len(centers) >= 5
        validation_cases.extend(validation)
    assert len(validation_cases) == len(set(validation_cases)) == 40
    occurrences = Counter(manifest["cases"][case]["center"] for case in validation_cases)
    assert occurrences == {1: 16, 2: 8, 3: 5, 4: 3, 6: 3, 7: 5}
