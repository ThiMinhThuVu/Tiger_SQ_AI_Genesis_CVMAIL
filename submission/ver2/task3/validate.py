"""Validate standalone Task 3 output."""
import argparse
import csv
from pathlib import Path

import numpy as np

from model import STATIONS


def validate_output(input_dir: Path, output_dir: Path) -> None:
    expected = {path.stem for path in input_dir.glob("*.png")}
    csv_path = output_dir / "task3.csv"
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["case_id", *STATIONS]:
            raise ValueError(f"Invalid Task 3 header: {reader.fieldnames}")
        rows = list(reader)
    ids = [row["case_id"] for row in rows]
    if not expected or len(ids) != len(expected) or set(ids) != expected:
        raise ValueError("task3.csv must contain exactly one row per input frame")
    for row in rows:
        for station in STATIONS:
            value = float(row[station])
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"Invalid probability for {row['case_id']} station {station}")
    extras = {path.name for path in output_dir.iterdir()} - {"task3.csv"}
    if extras:
        raise ValueError(f"Unexpected output entries: {sorted(extras)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    validate_output(args.input, args.output)
