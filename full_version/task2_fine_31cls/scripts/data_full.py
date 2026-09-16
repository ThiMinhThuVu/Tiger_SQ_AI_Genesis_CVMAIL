"""Manifest-backed full-data records without changing the historical 10-case split."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from tiger_models.data import LabelMap, Record, STATIONS, load_visibility


FRAME_RE = re.compile(
    r"^(?P<case_id>center_(?P<center>\d+)_case_(?P<case>\d+))_"
    r"(?P<station>6L|6R|7L|7R|8|9|10L|10R|11L|11R|12L|12R|13L|13R)"
    r"(?:_frame_(?P<frame>\d+))?\.png$"
)

DATASET517_REMOVED_FRAMES = (
    "center_1_case_3_13R.png",
    "center_1_case_10_6R.png",
    "center_1_case_11_6L.png",
    "center_1_case_12_6R.png",
    "center_1_case_15_6R.png",
    "center_7_case_2_13R.png",
    "center_7_case_3_12L.png",
)
DATASET517_CORRECTED_FRAME = "center_7_case_5_10L.png"
DATASET517_FINE_SHA256 = "e29be0a9e53a76720fceb12b752f3c590084102f019cc08eba85c418876dde39"
DATASET517_COARSE_SHA256 = "3ccee03bdae8ec7cf9ddceb1cd27c2b06da4f089b4bbfed94af58658422add81"

# Canonical seed-2026 outer folds, balanced to 8 cases and 102--106 frames.
TEST_CASES_SEED_2026 = (
    ("center_1_case_6", "center_1_case_8", "center_1_case_13", "center_2_case_7",
     "center_3_case_2", "center_4_case_2", "center_6_case_2", "center_7_case_5"),
    ("center_1_case_2", "center_1_case_14", "center_1_case_15", "center_2_case_4",
     "center_2_case_6", "center_3_case_3", "center_4_case_3", "center_7_case_6"),
    ("center_1_case_5", "center_1_case_11", "center_1_case_16", "center_1_case_17",
     "center_2_case_5", "center_3_case_5", "center_6_case_3", "center_7_case_4"),
    ("center_1_case_3", "center_1_case_9", "center_1_case_10", "center_2_case_3",
     "center_2_case_8", "center_3_case_6", "center_6_case_4", "center_7_case_3"),
    ("center_1_case_4", "center_1_case_7", "center_1_case_12", "center_2_case_2",
     "center_2_case_9", "center_3_case_4", "center_4_case_4", "center_7_case_2"),
)

# Four cases from the next outer fold: unique across folds and center-diverse.
VALIDATION_CASES_SEED_2026 = (
    ("center_1_case_2", "center_2_case_4", "center_3_case_3", "center_4_case_3"),
    ("center_1_case_5", "center_2_case_5", "center_3_case_5", "center_6_case_3"),
    ("center_1_case_3", "center_2_case_3", "center_3_case_6", "center_6_case_4"),
    ("center_1_case_4", "center_2_case_2", "center_3_case_4", "center_7_case_2"),
    ("center_1_case_6", "center_2_case_7", "center_3_case_2", "center_4_case_2"),
)


@dataclass(frozen=True)
class CaseInfo:
    case_id: str
    center: int
    frames: int
    width: int
    height: int

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"


def parse_frame_name(name: str) -> dict[str, str | int | None]:
    match = FRAME_RE.fullmatch(Path(name).name)
    if match is None:
        raise ValueError(f"Unrecognized TIGER frame name: {name}")
    fields: dict[str, str | int | None] = match.groupdict()
    fields["center"] = int(str(fields["center"]))
    fields["case"] = int(str(fields["case"]))
    fields["frame"] = int(str(fields["frame"])) if fields["frame"] is not None else None
    return fields


def _names_digest(names: list[str]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_cases(data_root: str | Path) -> tuple[dict[str, CaseInfo], dict[str, Any]]:
    root = Path(data_root)
    image_names = sorted(path.name for path in (root / "images").glob("*.png"))
    mask_names = sorted(path.name for path in (root / "masks_fine").glob("*.png"))
    coarse_mask_names = sorted(path.name for path in (root / "masks_coarse").glob("*.png"))
    if image_names != mask_names:
        raise ValueError(
            f"Image/fine-mask mismatch: only_images={sorted(set(image_names)-set(mask_names))[:10]}, "
            f"only_masks={sorted(set(mask_names)-set(image_names))[:10]}"
        )
    if image_names != coarse_mask_names:
        raise ValueError(
            f"Image/coarse-mask mismatch: only_images={sorted(set(image_names)-set(coarse_mask_names))[:10]}, "
            f"only_masks={sorted(set(coarse_mask_names)-set(image_names))[:10]}"
        )

    grouped: dict[str, list[str]] = {}
    for name in image_names:
        parsed = parse_frame_name(name)
        grouped.setdefault(str(parsed["case_id"]), []).append(name)

    cases: dict[str, CaseInfo] = {}
    for case_id, names in sorted(grouped.items()):
        sizes: set[tuple[int, int]] = set()
        for name in names:
            with Image.open(root / "images" / name) as handle:
                sizes.add(handle.size)
        if len(sizes) != 1:
            raise ValueError(f"Case {case_id} has multiple resolutions: {sorted(sizes)}")
        width, height = sizes.pop()
        center = int(parse_frame_name(names[0])["center"])
        cases[case_id] = CaseInfo(case_id, center, len(names), width, height)

    snapshot = {
        "case_count": len(cases),
        "frame_count": len(image_names),
        "image_names_sha256": _names_digest(image_names),
        "fine_mask_names_sha256": _names_digest(mask_names),
        "coarse_mask_names_sha256": _names_digest(coarse_mask_names),
    }
    corrected_fine = root / "masks_fine" / DATASET517_CORRECTED_FRAME
    corrected_coarse = root / "masks_coarse" / DATASET517_CORRECTED_FRAME
    if corrected_fine.is_file() and corrected_coarse.is_file():
        snapshot["corrected_fine_mask_sha256"] = _file_digest(corrected_fine)
        snapshot["corrected_coarse_mask_sha256"] = _file_digest(corrected_coarse)
    return cases, snapshot


def audit_fine_masks(data_root: str | Path, labelmap: LabelMap) -> dict[str, Any]:
    """Check all paired dimensions and RGB values without constructing tensors."""
    root = Path(data_root)
    allowed = {tuple(int(value) for value in rgb) for rgb in labelmap.fine_rgb}
    observed: set[tuple[int, int, int]] = set()
    checked = 0
    for image_path in sorted((root / "images").glob("*.png")):
        mask_path = root / "masks_fine" / image_path.name
        if not mask_path.is_file():
            raise FileNotFoundError(f"Missing fine mask for {image_path.name}")
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            if image.size != mask.size:
                raise ValueError(
                    f"Image/mask dimensions differ for {image_path.name}: "
                    f"{image.size} != {mask.size}"
                )
            colors = mask.convert("RGB").getcolors(maxcolors=256)
            if colors is None:
                raise ValueError(f"Mask has more than 256 RGB values: {mask_path}")
            frame_colors = {tuple(rgb) for _, rgb in colors}
            unknown = frame_colors - allowed
            if unknown:
                raise ValueError(f"Unknown fine-mask RGB values in {mask_path}: {sorted(unknown)[:10]}")
            observed.update(frame_colors)
            checked += 1
    missing_ids = [
        index for index, rgb in enumerate(labelmap.fine_rgb)
        if tuple(int(value) for value in rgb) not in observed
    ]
    return {
        "paired_frames_checked": checked,
        "observed_label_colors": len(observed),
        "allowed_label_colors": len(allowed),
        "unobserved_label_ids": missing_ids,
        "unobserved_label_names": [labelmap.fine_names[index] for index in missing_ids],
        "dimensions_match": True,
        "rgb_values_valid": True,
    }


def _split_summary(case_ids: list[str], cases: dict[str, CaseInfo]) -> dict[str, Any]:
    selected = [cases[case_id] for case_id in case_ids]
    return {
        "cases": len(selected),
        "frames": sum(case.frames for case in selected),
        "centers": sorted({case.center for case in selected}),
        "resolutions": sorted({case.resolution for case in selected}),
    }


def build_manifest(data_root: str | Path, split_seed: int = 2026) -> dict[str, Any]:
    if split_seed != 2026:
        raise ValueError("Only the reviewed immutable split_seed=2026 is defined")
    cases, snapshot = discover_cases(data_root)
    all_cases = set(cases)
    expected_test = {case for fold in TEST_CASES_SEED_2026 for case in fold}
    if all_cases != expected_test:
        raise ValueError(
            f"Dataset cases differ from reviewed split: missing={sorted(expected_test-all_cases)}, "
            f"unexpected={sorted(all_cases-expected_test)}"
        )

    folds: dict[str, Any] = {}
    validation_occurrences: list[str] = []
    test_occurrences: list[str] = []
    for fold, (test_tuple, validation_tuple) in enumerate(
        zip(TEST_CASES_SEED_2026, VALIDATION_CASES_SEED_2026)
    ):
        test_cases = sorted(test_tuple)
        validation_cases = sorted(validation_tuple)
        train_cases = sorted(all_cases - set(test_cases) - set(validation_cases))
        if set(test_cases) & set(validation_cases):
            raise ValueError(f"Fold {fold}: validation/test overlap")
        if (len(train_cases), len(validation_cases), len(test_cases)) != (28, 4, 8):
            raise ValueError(f"Fold {fold}: expected 28/4/8 cases")
        if len(_split_summary(validation_cases, cases)["centers"]) != 4:
            raise ValueError(f"Fold {fold}: validation must cover four centers")
        validation_occurrences.extend(validation_cases)
        test_occurrences.extend(test_cases)
        folds[str(fold)] = {
            "train_cases": train_cases,
            "validation_cases": validation_cases,
            "test_cases": test_cases,
            "summary": {
                "train": _split_summary(train_cases, cases),
                "validation": _split_summary(validation_cases, cases),
                "test": _split_summary(test_cases, cases),
            },
        }
    if len(set(test_occurrences)) != 40 or len(test_occurrences) != 40:
        raise ValueError("Each case must occur in outer test exactly once")
    if len(set(validation_occurrences)) != 20 or len(validation_occurrences) != 20:
        raise ValueError("The 20 validation case assignments must be unique")

    return {
        "version": "full40_case_folds_v1",
        "split_seed": split_seed,
        "grouping_unit": "case_id",
        "snapshot": snapshot,
        "cases": {case_id: asdict(info) | {"resolution": info.resolution}
                  for case_id, info in sorted(cases.items())},
        "folds": folds,
    }


def build_trainval_manifest(data_root: str | Path, split_seed: int = 2026) -> dict[str, Any]:
    """Build the reviewed center-stratified 5-fold, 32/8 train/val split.

    Centers 4 and 6 contain only three cases each, so they cannot occur in all
    five validation folds.  The canonical outer-fold assignment spreads each
    of those cases over three distinct folds, while centers 2, 3, and 7 occur
    in every fold.  Every case is validation exactly once.
    """
    if split_seed != 2026:
        raise ValueError("Only the reviewed immutable split_seed=2026 is defined")
    cases, snapshot = discover_cases(data_root)
    root = Path(data_root)
    active_names = {path.name for path in (root / "images").glob("*.png")}
    invalid = sorted(active_names.intersection(DATASET517_REMOVED_FRAMES))
    if invalid:
        raise ValueError(f"Removed dataset-517 frames are still active: {invalid}")
    if int(snapshot["frame_count"]) != 517:
        raise ValueError(f"Dataset-517 snapshot must contain 517 frames, got {snapshot['frame_count']}")
    if snapshot.get("corrected_fine_mask_sha256") != DATASET517_FINE_SHA256:
        raise ValueError("Dataset-517 corrected fine mask checksum does not match Synapse")
    if snapshot.get("corrected_coarse_mask_sha256") != DATASET517_COARSE_SHA256:
        raise ValueError("Dataset-517 corrected coarse mask checksum does not match Synapse")
    all_cases = set(cases)
    expected = {case for fold in TEST_CASES_SEED_2026 for case in fold}
    if all_cases != expected:
        raise ValueError(
            f"Dataset cases differ from reviewed split: missing={sorted(expected-all_cases)}, "
            f"unexpected={sorted(all_cases-expected)}"
        )
    folds: dict[str, Any] = {}
    occurrences: list[str] = []
    for fold, validation_tuple in enumerate(TEST_CASES_SEED_2026):
        validation_cases = sorted(validation_tuple)
        train_cases = sorted(all_cases - set(validation_cases))
        if (len(train_cases), len(validation_cases)) != (32, 8):
            raise ValueError(f"Fold {fold}: expected 32 train / 8 validation cases")
        val_centers = _split_summary(validation_cases, cases)["centers"]
        if not {1, 2, 3, 7}.issubset(val_centers) or len(val_centers) < 5:
            raise ValueError(f"Fold {fold}: insufficient center coverage: {val_centers}")
        occurrences.extend(validation_cases)
        folds[str(fold)] = {
            "train_cases": train_cases,
            "validation_cases": validation_cases,
            "summary": {
                "train": _split_summary(train_cases, cases),
                "validation": _split_summary(validation_cases, cases),
            },
        }
    if len(occurrences) != 40 or len(set(occurrences)) != 40:
        raise ValueError("Every case must occur in validation exactly once")
    return {
        "version": "full40_center_stratified_5fold_trainval_d517_v2",
        "split_seed": split_seed,
        "grouping_unit": "case_id",
        "split_policy": "32 train / 8 validation; no local test",
        "center_policy": "centers 2,3,7 in all folds; scarce centers 4,6 spread across three folds each",
        "snapshot": snapshot,
        "cases": {case_id: asdict(info) | {"resolution": info.resolution}
                  for case_id, info in sorted(cases.items())},
        "folds": folds,
    }


def load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def records_for_manifest(
    data_root: str | Path, manifest: dict[str, Any], fold: int, split: str
) -> list[Record]:
    if fold not in range(5):
        raise ValueError("fold must be in 0..4")
    canonical = {"val": "validation"}.get(split, split)
    if canonical not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    if f"{canonical}_cases" not in manifest["folds"][str(fold)]:
        raise ValueError(f"Split {canonical!r} is not defined by manifest {manifest.get('version')!r}")
    case_ids = set(manifest["folds"][str(fold)][f"{canonical}_cases"])
    root = Path(data_root)
    records: list[Record] = []
    for image in sorted((root / "images").glob("*.png")):
        case_id = str(parse_frame_name(image.name)["case_id"])
        if case_id not in case_ids:
            continue
        fine_mask = root / "masks_fine" / image.name
        if not fine_mask.is_file():
            raise FileNotFoundError(f"Missing fine mask for {image.name}")
        records.append(Record(
            image.name, case_id, image, fine_mask,
            np.zeros(len(STATIONS), dtype=np.float32),
        ))
    expected = manifest["folds"][str(fold)]["summary"][canonical]
    if len(records) != int(expected["frames"]):
        raise ValueError(f"Fold {fold} {canonical}: manifest/frame count mismatch")
    if len({record.case_id for record in records}) != int(expected["cases"]):
        raise ValueError(f"Fold {fold} {canonical}: manifest/case count mismatch")
    return records


def visibility_records_for_manifest(
    data_root: str | Path, manifest: dict[str, Any], fold: int, split: str
) -> list[Record]:
    """Return only manifest-split frames that have genuine Task-3 labels.

    The dataset-517 snapshot has segmentation labels for 517 frames. Task-3
    uses only active frames present in the visibility CSV; missing annotations
    are excluded rather than being silently interpreted as all-negative targets.
    """
    root = Path(data_root)
    visibility = load_visibility(root / "lymph_node_station_visibility.csv")
    records = records_for_manifest(root, manifest, fold, split)
    return [
        Record(record.name, record.case_id, record.image, record.fine_mask,
               visibility[record.name])
        for record in records if record.name in visibility
    ]
