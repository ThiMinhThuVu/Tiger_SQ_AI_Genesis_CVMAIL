#!/usr/bin/env python3
"""Run the unmodified official 3-task entry point on a small format fixture."""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tiger_models.data import STATIONS, records_for_fold, write_visibility_ground_truth
from tiger_models.runtime import atomic_json


def main() -> int:
    root = ROOT / "artifacts/new_baselines/evaluator_conformance"
    if root.exists():
        shutil.rmtree(root)
    gt = root / "input/gt"
    pred = root / "input/predictions/generated_fixture"
    for base in (gt, pred):
        (base / "task1").mkdir(parents=True)
        (base / "task2").mkdir(parents=True)
    records = records_for_fold(ROOT / "data", 0, "validation")
    for record in records:
        for task, folder in (("task1", "masks_fine"), ("task2", "masks_coarse")):
            image = Image.open(ROOT / "data" / folder / record.name).convert("RGB")
            image = image.resize((28, 16), Image.Resampling.NEAREST)
            image.save(gt / task / record.name)
            image.save(pred / task / record.name)
    write_visibility_ground_truth(records, gt / "task3.csv")
    with (pred / "task3.csv").open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["case_id", *STATIONS])
        for record in sorted(records, key=lambda item: item.name):
            writer.writerow([Path(record.name).stem, *[0.9 if value else 0.1 for value in record.visibility]])
    report = root / "official_report.md"
    evaluator = ROOT / "external/tigersqai_evaluation"
    command = [
        sys.executable, str(evaluator / "metrics/01_evaluate_challenge.py"),
        "--gt", str(gt), "--pred", str(root / "input/predictions"),
        "--out", str(report), "--figures-dir", "none", "--save-json",
    ]
    process = subprocess.run(command, cwd=evaluator, capture_output=True, text=True)
    (root / "official_evaluator.log").write_text(process.stdout + process.stderr)
    result = {
        "status": "PASS" if process.returncode == 0 and report.is_file() else "FAIL",
        "returncode": process.returncode,
        "mode": "downscaled generated format/API conformance fixture; not benchmark metrics",
        "frames": len(records), "tasks": [1, 2, 3],
        "official_evaluator_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=evaluator, text=True
        ).strip(),
    }
    atomic_json(root / "result.json", result)
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise RuntimeError(f"Official evaluator fixture failed; see {root/'official_evaluator.log'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
