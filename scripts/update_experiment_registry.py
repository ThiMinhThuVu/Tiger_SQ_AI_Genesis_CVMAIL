#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tiger_models.runtime import update_registry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    parser.add_argument("--status", default=None)
    parser.add_argument("--last-completed-phase", default=None)
    parser.add_argument("--completed-folds", default=None, help="JSON list")
    parser.add_argument("--failure-reason", default=None)
    args = parser.parse_args()
    fields = {
        key: value for key, value in {
            "status": args.status,
            "last_completed_phase": args.last_completed_phase,
            "failure_reason": args.failure_reason,
        }.items() if value is not None
    }
    if args.completed_folds is not None:
        fields["completed_folds"] = json.loads(args.completed_folds)
    update_registry(ROOT / "configs/experiment_registry.yaml", args.experiment, **fields)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
