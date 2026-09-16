from pathlib import Path

from full_version.task2_fine_31cls.scripts.data_full import (
    audit_fine_masks,
    build_manifest,
    parse_frame_name,
    records_for_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"


def test_parser_handles_repeated_station_frames():
    parsed = parse_frame_name("center_2_case_9_10R_frame_2.png")
    assert parsed["case_id"] == "center_2_case_9"
    assert parsed["station"] == "10R"
    assert parsed["frame"] == 2


def test_manifest_has_grouped_balanced_folds():
    manifest = build_manifest(DATA_ROOT)
    assert manifest["snapshot"]["case_count"] == 40
    assert manifest["snapshot"]["frame_count"] == 517
    test_cases = []
    validation_cases = []
    for payload in manifest["folds"].values():
        train = set(payload["train_cases"])
        validation = set(payload["validation_cases"])
        test = set(payload["test_cases"])
        assert (len(train), len(validation), len(test)) == (28, 4, 8)
        assert not train & validation
        assert not train & test
        assert not validation & test
        assert payload["summary"]["validation"]["cases"] == 4
        assert len(payload["summary"]["validation"]["centers"]) == 4
        test_cases.extend(test)
        validation_cases.extend(validation)
    assert len(test_cases) == len(set(test_cases)) == 40
    assert len(validation_cases) == len(set(validation_cases)) == 20


def test_records_match_manifest_counts_and_keep_special_names():
    manifest = build_manifest(DATA_ROOT)
    all_names = set()
    for split in ("train", "validation", "test"):
        records = records_for_manifest(DATA_ROOT, manifest, 4, split)
        summary = manifest["folds"]["4"]["summary"][split]
        assert len(records) == summary["frames"]
        assert len({record.case_id for record in records}) == summary["cases"]
        all_names.update(record.name for record in records)
    assert len(all_names) == 517
    assert "center_7_case_3_12L.png" not in all_names
    assert any("_frame_" in name for name in all_names)


def test_mask_audit_on_one_temporary_pair(tmp_path):
    from PIL import Image
    from tiger_models.data import LabelMap

    (tmp_path / "images").mkdir()
    (tmp_path / "masks_fine").mkdir()
    name = "center_1_case_2_6L.png"
    Image.new("RGB", (4, 3), (1, 2, 3)).save(tmp_path / "images" / name)
    labelmap = LabelMap.load(DATA_ROOT / "labelmap.csv")
    Image.new("RGB", (4, 3), tuple(labelmap.fine_rgb[0])).save(tmp_path / "masks_fine" / name)
    result = audit_fine_masks(tmp_path, labelmap)
    assert result["paired_frames_checked"] == 1
    assert result["observed_label_colors"] == 1
    assert len(result["unobserved_label_ids"]) == 30
