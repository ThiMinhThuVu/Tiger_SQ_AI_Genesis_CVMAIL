#!/usr/bin/env python3
"""Evaluate best checkpoints on the 8-case validation folds and report centers."""
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

from full_version.task1_merged_16cls.scripts.train_coarse_only import (
    build_model as build_coarse_model, evaluate_model as evaluate_coarse_model,
    loader as coarse_loader,
)
from full_version.task2_fine_31cls.scripts.data_full import (
    build_trainval_manifest, load_manifest, records_for_manifest,
)
from scripts.infer_mask2former_improved_test import infer_fold
from tiger_models.data import LabelMap


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("task1", "task2"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    return parser.parse_args()


def center(case_id: str) -> str:
    return case_id.split("_case_", 1)[0]


def mean(values) -> float:
    return float(statistics.mean(values))


def summarize_centers(case_metrics: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    output = {}
    for center_id in sorted({center(case_id) for case_id in case_metrics}):
        rows = [row for case_id, row in case_metrics.items() if center(case_id) == center_id]
        metrics = {key: mean(row[key] for row in rows) for key in rows[0]}
        output[center_id] = {"cases": len(rows), **metrics}
    return output


def main() -> None:
    args = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    config = yaml.safe_load(args.config.read_text())
    manifest = load_manifest(args.manifest)
    expected = build_trainval_manifest(args.data_root, int(config["split_seed"]))
    if manifest != expected:
        raise ValueError("Manifest is not the reviewed center-stratified dataset snapshot")
    labelmap = LabelMap.load(args.data_root / "labelmap.csv")
    folds, case_metrics = [], {}
    for fold in range(5):
        validation_cases = manifest["folds"][str(fold)]["validation_cases"]
        if args.task == "task2":
            def provider(data_root, current_fold, split):
                mapped = "validation" if split == "test" else split
                return records_for_manifest(data_root, manifest, current_fold, mapped)
            result = infer_fold(config, args.model_dir, fold, args.data_root,
                                torch.device("cuda"), labelmap, records_provider=provider)
            if result is None:
                raise FileNotFoundError(args.model_dir / f"fold_{fold}/best.pt")
            domains = ("fine", "coarse")
            for case_id in validation_cases:
                case_metrics[case_id] = {
                    "fine_dice": float(result["fine"]["per_case_dice"][case_id]),
                    "fine_nhd": float(result["fine"]["per_case_nhd"][case_id]),
                    "coarse_dice": float(result["coarse"]["per_case_dice"][case_id]),
                    "coarse_nhd": float(result["coarse"]["per_case_nhd"][case_id]),
                }
            keep = {key: result[key] for key in ("task1_score", "task12_surrogate",
                    "fine_dice", "fine_nhd", "coarse_dice", "coarse_nhd")}
        else:
            shim = argparse.Namespace(**vars(args), split_manifest=args.manifest,
                                      output_root=args.model_dir.parent)
            checkpoint = args.model_dir / f"fold_{fold}/best.pt"
            model = build_coarse_model(config)
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"])
            model.cuda()
            result = evaluate_coarse_model(model, coarse_loader(
                shim, config, manifest, labelmap, fold, "validation"),
                torch.device("cuda"), labelmap, True)
            for case_id in validation_cases:
                case_metrics[case_id] = {
                    "dice": float(result["per_case_dice"][case_id]),
                    "nhd": float(result["per_case_nhd"][case_id]),
                }
            keep = {key: result[key] for key in ("dice", "nhd", "task_score")}
        folds.append({"fold": fold, "validation_cases": validation_cases,
                      "validation_centers": manifest["folds"][str(fold)]["summary"]["validation"]["centers"],
                      **keep})
    centers = summarize_centers(case_metrics)
    score_key = "fine_dice" if args.task == "task2" else "dice"
    macro_center = {key: mean(row[key] for row in centers.values())
                    for key in next(iter(centers.values())) if key != "cases"}
    worst = min(centers, key=lambda key: centers[key][score_key])
    output = {
        "task": args.task, "split": "5_fold_oof_validation_no_local_test",
        "manifest_version": manifest["version"], "checkpoint_policy": "best validation checkpoint",
        "folds": folds, "per_case": case_metrics, "per_center": centers,
        "macro_over_centers": macro_center,
        "worst_center": {"center": worst, **centers[worst]},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "macro_over_centers": macro_center,
                      "worst_center": output["worst_center"]}, indent=2))


if __name__ == "__main__":
    main()
