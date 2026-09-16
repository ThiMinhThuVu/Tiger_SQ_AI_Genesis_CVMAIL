#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tiger_models.data import audit_dataset, save_audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--full", action="store_true", help="decode every mask pixel")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/new_baselines/data_audit.json")
    args = parser.parse_args()
    result = audit_dataset(args.data_root, full=args.full)
    save_audit(result, args.output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
