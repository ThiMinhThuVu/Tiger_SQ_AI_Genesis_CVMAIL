#!/usr/bin/env python3
"""Standalone generator for the 32/8 train-val-only split manifest.

Written outside full_version/ because hieu.ph has no write permission
inside that tree (owned by thu.vtm:thu.vtm, mode 0775). This script only
*imports* (read-only) from full_version/task2_fine_31cls/scripts/data_full.py
and writes its output here, in patch_hieu_trainval_split/.

`build_trainval_manifest` is a drop-in duplicate of the function proposed
for data_full.py (see data_full_ADDITION.py in this same directory) -- once
someone with write access appends that function to data_full.py, this
generator becomes redundant and full_version/.../prepare_trainval_splits.py
(also drafted here) should be used instead.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from full_version.task2_fine_31cls.scripts.data_full import (  # noqa: E402
    TEST_CASES_SEED_2026,
    _split_summary,
    discover_cases,
)


def build_trainval_manifest(data_root: str | Path, split_seed: int = 2026) -> dict[str, Any]:
    if split_seed != 2026:
        raise ValueError("Only the reviewed immutable split_seed=2026 is defined")
    cases, snapshot = discover_cases(data_root)
    all_cases = set(cases)
    expected = {case for fold in TEST_CASES_SEED_2026 for case in fold}
    if all_cases != expected:
        raise ValueError(
            f"Dataset cases differ from reviewed split: missing={sorted(expected - all_cases)}, "
            f"unexpected={sorted(all_cases - expected)}"
        )

    folds: dict[str, Any] = {}
    validation_occurrences: list[str] = []
    for fold, validation_tuple in enumerate(TEST_CASES_SEED_2026):
        validation_cases = sorted(validation_tuple)
        train_cases = sorted(all_cases - set(validation_cases))
        if (len(train_cases), len(validation_cases)) != (32, 8):
            raise ValueError(f"Fold {fold}: expected 32/8 cases")
        if len(_split_summary(validation_cases, cases)["centers"]) < 4:
            raise ValueError(f"Fold {fold}: validation must cover at least four centers")
        validation_occurrences.extend(validation_cases)
        folds[str(fold)] = {
            "train_cases": train_cases,
            "validation_cases": validation_cases,
            "summary": {
                "train": _split_summary(train_cases, cases),
                "validation": _split_summary(validation_cases, cases),
            },
        }
    if len(set(validation_occurrences)) != 40 or len(validation_occurrences) != 40:
        raise ValueError("Each case must occur in validation exactly once")

    return {
        "version": "full40_case_folds_v2_trainval",
        "split_seed": split_seed,
        "grouping_unit": "case_id",
        "derived_from": "full40_case_folds_v1 outer-test partition (TEST_CASES_SEED_2026)",
        "snapshot": snapshot,
        "cases": {case_id: asdict(info) | {"resolution": info.resolution}
                  for case_id, info in sorted(cases.items())},
        "folds": folds,
    }


def main() -> None:
    manifest = build_trainval_manifest(ROOT / "data", 2026)
    out = Path(__file__).parent / "case_folds_v2_trainval.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "output": str(out),
        "version": manifest["version"],
        "snapshot": manifest["snapshot"],
        "folds": {fold: payload["summary"] for fold, payload in manifest["folds"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
