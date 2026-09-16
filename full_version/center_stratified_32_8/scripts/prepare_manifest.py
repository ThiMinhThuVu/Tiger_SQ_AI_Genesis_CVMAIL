#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from full_version.task2_fine_31cls.scripts.data_full import build_trainval_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "splits/case_folds_d517.json")
    args = parser.parse_args()
    payload = build_trainval_manifest(args.data_root, 2026)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "folds": {
        fold: row["summary"] for fold, row in payload["folds"].items()
    }}, indent=2))


if __name__ == "__main__":
    main()
