#!/usr/bin/env python3
"""Audit the full dataset and write the immutable 32/8 train-val manifest.

Drop this into full_version/task2_fine_31cls/scripts/ (mirrors
prepare_splits.py, but calls build_trainval_manifest instead of
build_manifest). Requires data_full.py to already carry
build_trainval_manifest (see data_full_ADDITION.py / data_full.py.patch).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from full_version.task2_fine_31cls.scripts.data_full import audit_fine_masks, build_trainval_manifest
from tiger_models.data import LabelMap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "full_version/task2_fine_31cls/splits/case_folds_v2_trainval.json",
    )
    parser.add_argument("--split-seed", type=int, default=2026)
    args = parser.parse_args()

    manifest = build_trainval_manifest(args.data_root, args.split_seed)
    mask_audit = audit_fine_masks(args.data_root, LabelMap.load(args.data_root / "labelmap.csv"))
    manifest["fine_mask_audit"] = mask_audit
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "version": manifest["version"],
        "snapshot": manifest["snapshot"],
        "fine_mask_audit": mask_audit,
        "folds": {fold: payload["summary"] for fold, payload in manifest["folds"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
