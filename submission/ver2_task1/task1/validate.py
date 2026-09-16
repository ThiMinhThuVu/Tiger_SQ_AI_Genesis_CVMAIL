"""Validate standalone Task 1 output."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from model import TASK1_PALETTE


def _packed(array: np.ndarray) -> np.ndarray:
    return (
        (array[..., 0].astype(np.uint32) << 16)
        | (array[..., 1].astype(np.uint32) << 8)
        | array[..., 2].astype(np.uint32)
    )


def _validate_mask(path: Path, expected_size: tuple[int, int]) -> None:
    with Image.open(path) as handle:
        if handle.format != "PNG" or handle.mode != "RGB":
            raise ValueError(f"Task 1 mask must be an RGB PNG: {path}")
        if handle.size != expected_size:
            raise ValueError(f"Mask size {handle.size} differs from input {expected_size}: {path}")
        array = np.asarray(handle)
    allowed = set(int(value) for value in _packed(TASK1_PALETTE.reshape(-1, 1, 3)).reshape(-1))
    unknown = set(int(value) for value in np.unique(_packed(array))) - allowed
    if unknown:
        raise ValueError(f"Mask contains {len(unknown)} unrecognised RGB colours: {path}")


def validate_output(input_dir: Path, output_dir: Path) -> None:
    inputs = sorted(input_dir.glob("*.png"))
    expected = {path.name for path in inputs}
    actual = {path.name for path in (output_dir / "task1").glob("*.png")}
    if not inputs or actual != expected:
        raise ValueError(
            f"Task 1 filenames differ: missing={sorted(expected-actual)[:5]}, "
            f"extra={sorted(actual-expected)[:5]}"
        )
    for input_path in inputs:
        with Image.open(input_path) as image:
            size = image.size
        _validate_mask(output_dir / "task1" / input_path.name, size)
    extras = {path.name for path in output_dir.iterdir()} - {"task1"}
    if extras:
        raise ValueError(f"Unexpected output entries: {sorted(extras)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    validate_output(args.input, args.output)
