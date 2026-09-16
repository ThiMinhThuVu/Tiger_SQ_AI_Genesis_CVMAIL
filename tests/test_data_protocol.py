from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from tiger_models.data import (
    STATIONS, TEST_CASES, VALIDATION_CASES, LabelMap, audit_dataset, load_visibility,
    records_for_fold,
)


ROOT = Path(__file__).resolve().parents[1]


def test_dataset_gate_and_hashes() -> None:
    result = audit_dataset(ROOT / "data", full=False)
    assert result["status"] == "PASS"
    assert result["frames"] == 140
    assert result["cases"] == 10
    assert result["visibility_dimension"] == 14


def test_rgb_roundtrip_and_fine_to_coarse() -> None:
    labels = LabelMap.load(ROOT / "data/labelmap.csv")
    name = "center_1_case_10_9.png"
    fine_rgb = np.asarray(Image.open(ROOT / "data/masks_fine" / name).convert("RGB"))
    coarse_rgb = np.asarray(Image.open(ROOT / "data/masks_coarse" / name).convert("RGB"))
    fine = labels.decode_fine(fine_rgb)
    coarse = labels.decode_coarse(coarse_rgb)
    assert np.array_equal(labels.encode_fine(fine), fine_rgb)
    assert np.array_equal(labels.encode_coarse(coarse), coarse_rgb)
    assert np.array_equal(labels.fine_to_coarse[fine], coarse)


def test_visibility_is_required_multihot_order() -> None:
    visibility = load_visibility(ROOT / "data/lymph_node_station_visibility.csv")
    vector = visibility["center_1_case_13_10L.png"]
    assert vector.shape == (14,)
    assert tuple(np.asarray(STATIONS)[vector.astype(bool)]) == ("6L", "7L", "8", "10L")


def test_case_folds_have_no_case_or_frame_leakage() -> None:
    for fold in range(5):
        train = records_for_fold(ROOT / "data", fold, "train")
        validation = records_for_fold(ROOT / "data", fold, "validation")
        test = records_for_fold(ROOT / "data", fold, "test")
        assert len(train) == 98 and len(validation) == 14 and len(test) == 28
        assert len({r.case_id for r in train}) == 7
        assert {r.case_id for r in validation} == set(VALIDATION_CASES[fold])
        assert {r.case_id for r in test} == set(TEST_CASES[fold])
        splits = (train, validation, test)
        for left in range(3):
            for right in range(left + 1, 3):
                assert not ({r.case_id for r in splits[left]} & {r.case_id for r in splits[right]})
                assert not ({r.name for r in splits[left]} & {r.name for r in splits[right]})
