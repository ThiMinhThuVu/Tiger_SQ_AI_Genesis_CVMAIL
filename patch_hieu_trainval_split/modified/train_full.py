#!/usr/bin/env python3
"""Train the 31-ID model with the full-data case manifest."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from full_version.task2_fine_31cls.scripts.data_full import (
    build_manifest,
    build_trainval_manifest,
    load_manifest,
    records_for_manifest,
)

_MANIFEST_BUILDERS = {
    "full40_case_folds_v1": build_manifest,
    "full40_case_folds_v2_trainval": build_trainval_manifest,
}
from scripts.train_mask2former_improved import run_fold
from tiger_models.data import LabelMap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--run-all-folds", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--output-root", type=Path,
        default=ROOT / "full_version/task2_fine_31cls/artifacts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    manifest = load_manifest(args.split_manifest)
    builder = _MANIFEST_BUILDERS.get(manifest.get("version"))
    if builder is None:
        raise ValueError(f"Unsupported split manifest: {manifest.get('version')}")
    if int(manifest.get("split_seed", -1)) != int(config["split_seed"]):
        raise ValueError("Config split_seed does not match manifest")
    current = builder(args.data_root, int(config["split_seed"]))
    if current["snapshot"] != manifest.get("snapshot"):
        raise ValueError("Current dataset snapshot does not match the immutable split manifest")
    labelmap = LabelMap.load(args.data_root / "labelmap.csv")
    config["fine_to_coarse"] = labelmap.fine_to_coarse.astype(int).tolist()
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA unavailable; use --allow-cpu only for a deliberate CPU smoke run")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def provider(data_root: Path, fold: int, split: str):
        return records_for_manifest(data_root, manifest, fold, split)

    folds = range(5) if args.run_all_folds else [args.fold]
    for fold in folds:
        out = args.output_root / config["experiment_id"] / f"fold_{fold}"
        if (out / "completion.json").exists():
            raise FileExistsError(f"Refusing to overwrite completed run: {out}")
        out.mkdir(parents=True, exist_ok=True)
        (out / "run_provenance.json").write_text(json.dumps({
            "config": str(args.config.resolve()),
            "split_manifest": str(args.split_manifest.resolve()),
            "manifest_version": manifest["version"],
            "dataset_snapshot": manifest["snapshot"],
            "base_seed": int(config["seed"]),
            "effective_seed": int(config["seed"]) + fold,
            "fold": fold,
            "smoke": bool(args.smoke),
        }, indent=2) + "\n")
        run_fold(config, fold, args, labelmap, device, records_provider=provider)


if __name__ == "__main__":
    main()
