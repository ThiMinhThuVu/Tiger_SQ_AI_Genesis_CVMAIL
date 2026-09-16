from __future__ import annotations

import csv
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

SUBMISSION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUBMISSION_DIR))

from contract import validate_output
from model import STATIONS, TASK1_PALETTE, TASK2_PALETTE


def _valid_tree(root: Path) -> tuple[Path, Path]:
    input_dir = root / "input"
    output_dir = root / "output"
    (output_dir / "task1").mkdir(parents=True)
    (output_dir / "task2").mkdir(parents=True)
    input_dir.mkdir()
    name = "center_x_case_y_6R.png"
    Image.fromarray(np.full((7, 11, 3), 127, dtype=np.uint8)).save(input_dir / name)
    Image.fromarray(np.tile(TASK1_PALETTE[3], (7, 11, 1))).save(output_dir / "task1" / name)
    Image.fromarray(np.tile(TASK2_PALETTE[22], (7, 11, 1))).save(output_dir / "task2" / name)
    with (output_dir / "task3.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case_id", *STATIONS])
        writer.writerow([Path(name).stem, *([0.25] * len(STATIONS))])
    return input_dir, output_dir


def test_valid_combined_output(tmp_path: Path) -> None:
    input_dir, output_dir = _valid_tree(tmp_path)
    validate_output(input_dir, output_dir)


def test_task1_rejects_fine_only_colour(tmp_path: Path) -> None:
    input_dir, output_dir = _valid_tree(tmp_path)
    name = next(input_dir.glob("*.png")).name
    Image.fromarray(np.tile(TASK2_PALETTE[2], (7, 11, 1))).save(output_dir / "task1" / name)
    with pytest.raises(ValueError, match="unrecognised RGB"):
        validate_output(input_dir, output_dir)


def test_task3_rejects_binary_contract_error(tmp_path: Path) -> None:
    input_dir, output_dir = _valid_tree(tmp_path)
    rows = (output_dir / "task3.csv").read_text().splitlines()
    rows[1] = rows[1].replace("0.25", "1.2", 1)
    (output_dir / "task3.csv").write_text("\n".join(rows) + "\n")
    with pytest.raises(ValueError, match=r"outside \[0,1\]"):
        validate_output(input_dir, output_dir)


def test_mask_helper_import_does_not_require_combined_model_symbols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Standalone Task 2 must not need Task 1/3 exports at import time."""
    standalone_model = types.ModuleType("model")
    standalone_model.TASK2_PALETTE = TASK2_PALETTE
    monkeypatch.setitem(sys.modules, "model", standalone_model)
    spec = importlib.util.spec_from_file_location(
        "standalone_contract", SUBMISSION_DIR / "contract.py"
    )
    assert spec is not None and spec.loader is not None
    standalone_contract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(standalone_contract)
    assert callable(standalone_contract._validate_mask)
