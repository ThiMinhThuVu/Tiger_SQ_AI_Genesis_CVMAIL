#!/usr/bin/env python3
"""Per-fold inference and exact case-level OOF aggregation for full40 fine-31."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from full_version.task2_fine_31cls.scripts.data_full import (
    build_manifest,
    load_manifest,
    records_for_manifest,
)
from scripts.infer_mask2former_improved_test import infer_fold
from tiger_models.data import LabelMap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("fold", "aggregate"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    return parser.parse_args()


def checked_context(args: argparse.Namespace):
    config = yaml.safe_load(args.config.read_text())
    manifest = load_manifest(args.split_manifest)
    current = build_manifest(args.data_root, int(config["split_seed"]))
    if manifest.get("version") != "full40_case_folds_v1":
        raise ValueError(f"Unsupported manifest version: {manifest.get('version')}")
    if manifest.get("snapshot") != current.get("snapshot"):
        raise ValueError("Current dataset does not match the immutable split manifest")
    return config, manifest


def write_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing evaluation: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def evaluate_one_fold(args: argparse.Namespace) -> None:
    if args.fold is None:
        raise ValueError("--fold is required in fold mode")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for held-out test inference")
    config, manifest = checked_context(args)
    path = args.output_dir / "folds" / f"fold_{args.fold}.json"
    expected = set(manifest["folds"][str(args.fold)]["test_cases"])
    if path.is_file():
        existing = json.loads(path.read_text())
        reusable = (
            int(existing.get("fold", -1)) == args.fold
            and set(existing.get("test_cases", [])) == expected
            and existing.get("manifest_version") == manifest["version"]
            and existing.get("dataset_snapshot") == manifest["snapshot"]
            and existing.get("split") == "held_out_local_test"
        )
        if not reusable:
            raise ValueError(f"Existing fold evaluation is not reusable: {path}")
        print(json.dumps({
            "output": str(path), "fold": args.fold, "status": "reused",
            "task1_score": existing["task1_score"],
            "fine_dice": existing["fine_dice"], "fine_nhd": existing["fine_nhd"],
        }, indent=2))
        return
    labelmap = LabelMap.load(args.data_root / "labelmap.csv")

    def provider(data_root: Path, fold: int, split: str):
        return records_for_manifest(data_root, manifest, fold, split)

    result = infer_fold(
        config, args.model_dir, args.fold, args.data_root,
        torch.device("cuda"), labelmap, records_provider=provider,
    )
    if result is None:
        raise FileNotFoundError(f"Missing best checkpoint for fold {args.fold}")
    if set(result["test_cases"]) != expected:
        raise ValueError(f"Fold {args.fold} evaluated the wrong held-out cases")
    result.update({
        "manifest_version": manifest["version"],
        "dataset_snapshot": manifest["snapshot"],
        "split": "held_out_local_test",
        "selection_policy": "best checkpoint selected on validation task1_score only",
    })
    write_new(path, result)
    print(json.dumps({
        "output": str(path), "fold": args.fold,
        "task1_score": result["task1_score"],
        "fine_dice": result["fine_dice"], "fine_nhd": result["fine_nhd"],
    }, indent=2))


def _case_values(folds: list[dict], domain: str, metric: str) -> dict[str, float]:
    merged: dict[str, float] = {}
    key = f"per_case_{metric}"
    for fold in folds:
        for case_id, value in fold[domain][key].items():
            if case_id in merged:
                raise ValueError(f"OOF case occurs more than once: {case_id}")
            merged[case_id] = float(value)
    return merged


def _score(dice: float, nhd: float) -> float:
    return (dice + 1.0 - nhd) / 2.0


def aggregate(args: argparse.Namespace) -> None:
    config, manifest = checked_context(args)
    folds = []
    for fold in range(5):
        path = args.output_dir / "folds" / f"fold_{fold}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing fold evaluation: {path}")
        payload = json.loads(path.read_text())
        if int(payload["fold"]) != fold:
            raise ValueError(f"Wrong fold payload in {path}")
        folds.append(payload)

    names = [name for fold in folds for name in fold["names"]]
    cases = [case for fold in folds for case in fold["test_cases"]]
    if len(names) != len(set(names)) or len(names) != int(manifest["snapshot"]["frame_count"]):
        raise ValueError("OOF frame coverage must contain every frame exactly once")
    if len(cases) != len(set(cases)) or len(cases) != int(manifest["snapshot"]["case_count"]):
        raise ValueError("OOF case coverage must contain every case exactly once")

    fine_dice = _case_values(folds, "fine", "dice")
    fine_nhd = _case_values(folds, "fine", "nhd")
    coarse_dice = _case_values(folds, "coarse", "dice")
    coarse_nhd = _case_values(folds, "coarse", "nhd")
    oof = {
        "cases": len(cases), "frames": len(names),
        "fine_dice": statistics.mean(fine_dice.values()),
        "fine_nhd": statistics.mean(fine_nhd.values()),
        "coarse_dice": statistics.mean(coarse_dice.values()),
        "coarse_nhd": statistics.mean(coarse_nhd.values()),
    }
    oof["task1_score"] = _score(oof["fine_dice"], oof["fine_nhd"])
    oof["task12_surrogate"] = statistics.mean((
        oof["fine_dice"], 1.0 - oof["fine_nhd"],
        oof["coarse_dice"], 1.0 - oof["coarse_nhd"],
    ))

    centers: dict[str, dict] = {}
    center_ids = sorted({case_id.split("_case_")[0] for case_id in cases})
    for center_id in center_ids:
        selected = [case_id for case_id in cases if case_id.startswith(center_id + "_case_")]
        fd = statistics.mean(fine_dice[case_id] for case_id in selected)
        fn = statistics.mean(fine_nhd[case_id] for case_id in selected)
        centers[center_id] = {
            "cases": len(selected), "fine_dice": fd, "fine_nhd": fn,
            "task1_score": _score(fd, fn),
        }
    worst_center = min(centers, key=lambda center: centers[center]["task1_score"])
    metric_keys = ("task1_score", "task12_surrogate", "fine_dice", "fine_nhd",
                   "coarse_dice", "coarse_nhd")
    fold_summary = {
        key: {
            "mean": statistics.mean(fold[key] for fold in folds),
            "sample_standard_deviation": statistics.stdev(fold[key] for fold in folds),
        }
        for key in metric_keys
    }
    output = {
        "experiment": config["experiment_id"],
        "manifest_version": manifest["version"],
        "dataset_snapshot": manifest["snapshot"],
        "split": "40_case_out_of_fold_held_out_local_test",
        "checkpoint_selection_policy": "validation task1_score only",
        "folds": folds,
        "fold_summary": fold_summary,
        "oof_case_aggregate": oof,
        "per_center": centers,
        "worst_center": {"center": worst_center, **centers[worst_center]},
        "interpretation_boundary": "Local held-out OOF result; not hidden challenge evaluation.",
    }
    path = args.output_dir / "oof_test_metrics.json"
    write_new(path, output)
    print(json.dumps({
        "output": str(path), "oof_case_aggregate": oof,
        "worst_center": output["worst_center"], "fold_summary": fold_summary,
    }, indent=2))


def main() -> None:
    args = parse_args()
    if args.mode == "fold":
        evaluate_one_fold(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
