"""Validation for the current TIGER SQ-AI task123 output contract."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

def _packed(array: np.ndarray) -> np.ndarray:
    return (
        (array[..., 0].astype(np.uint32) << 16)
        | (array[..., 1].astype(np.uint32) << 8)
        | array[..., 2].astype(np.uint32)
    )


def _allowed_colours(palette: np.ndarray) -> set[int]:
    return set(int(value) for value in _packed(palette.reshape(-1, 1, 3)).reshape(-1))


def _validate_mask(path: Path, expected_size: tuple[int, int], palette: np.ndarray) -> None:
    with Image.open(path) as handle:
        if handle.format != "PNG":
            raise ValueError(f"Mask is not a PNG: {path}")
        if handle.mode != "RGB":
            raise ValueError(f"Mask must be RGB, got {handle.mode}: {path}")
        if handle.size != expected_size:
            raise ValueError(
                f"Mask size {handle.size} differs from input {expected_size}: {path}"
            )
        array = np.asarray(handle)
    unknown = set(int(value) for value in np.unique(_packed(array))) - _allowed_colours(palette)
    if unknown:
        raise ValueError(f"Mask contains {len(unknown)} unrecognised RGB colours: {path}")


def validate_output(input_dir: Path, output_dir: Path) -> None:
    # Combined-output symbols are imported lazily so standalone Task 2 can
    # reuse _validate_mask without requiring Task 1 or Task 3 model exports.
    from model import STATIONS, TASK1_PALETTE, TASK2_PALETTE

    inputs = sorted(input_dir.glob("*.png"))
    if not inputs:
        raise ValueError(f"No PNG input frames found in {input_dir}")
    names = [path.name for path in inputs]
    expected_names = set(names)

    for task_name, palette in (("task1", TASK1_PALETTE), ("task2", TASK2_PALETTE)):
        task_dir = output_dir / task_name
        actual_names = {path.name for path in task_dir.glob("*.png")}
        if actual_names != expected_names:
            missing = sorted(expected_names - actual_names)
            extra = sorted(actual_names - expected_names)
            raise ValueError(
                f"{task_name} filename mismatch: missing={missing[:5]}, extra={extra[:5]}"
            )
        for input_path in inputs:
            with Image.open(input_path) as handle:
                expected_size = handle.size
            _validate_mask(task_dir / input_path.name, expected_size, palette)

    csv_path = output_dir / "task3.csv"
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected_header = ["case_id", *STATIONS]
        if reader.fieldnames != expected_header:
            raise ValueError(
                f"task3.csv header must be {expected_header}, got {reader.fieldnames}"
            )
        rows = list(reader)
    stems = [row["case_id"] for row in rows]
    expected_stems = {Path(name).stem for name in names}
    if len(stems) != len(expected_stems) or set(stems) != expected_stems:
        raise ValueError("task3.csv must contain exactly one row per input filename stem")
    for row in rows:
        for station in STATIONS:
            try:
                score = float(row[station])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid Task 3 score for {row['case_id']} station {station}"
                ) from exc
            if not np.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"Task 3 score outside [0,1] for {row['case_id']} station {station}"
                )

    allowed_entries = {"task1", "task2", "task3.csv"}
    extras = {path.name for path in output_dir.iterdir()} - allowed_entries
    if extras:
        raise ValueError(f"Unexpected output entries: {sorted(extras)}")
