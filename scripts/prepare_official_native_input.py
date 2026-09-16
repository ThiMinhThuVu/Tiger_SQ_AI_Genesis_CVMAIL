#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from validate_predictions import prepare_official_input


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    gt, predictions = prepare_official_input(
        args.prediction_dir, args.data_root, args.fold, args.destination,
        fixture=False, split=args.split,
    )
    print(f"gt={gt}")
    print(f"predictions={predictions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
