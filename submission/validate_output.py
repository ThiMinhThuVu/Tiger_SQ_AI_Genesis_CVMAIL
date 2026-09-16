#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from contract import validate_output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validate_output(args.input, args.output)
    print("PASS: output satisfies the TIGER SQ-AI task123 I/O contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
