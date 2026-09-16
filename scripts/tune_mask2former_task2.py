#!/usr/bin/env python3
"""Tune Task-2 coarse-mask post-processing for Mask2Former Swin-Tiny.

Each outer fold selects one conservative connected-component rule using only
its validation case, freezes the rule, and evaluates the two held-out test
cases. Baseline coarse IDs are exactly fine argmax mapped through labelmap.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from scipy.ndimage import label as connected_components
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_mask2former import build_model, collate, semantic_logits
from tiger_models.data import LabelMap, TigerMultitaskDataset, records_for_fold
from tiger_models.metrics import segmentation_metrics


VERSION = "mask2former_swin_tiny_task2_tuned_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/mask2former_swin_tiny.yaml")
    parser.add_argument("--model-dir", type=Path,
                        default=ROOT / "artifacts/mask2former/mask2former_swin_tiny")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "artifacts/mask2former" / VERSION)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--minimum-validation-gain", type=float, default=0.001)
    return parser.parse_args()


def task2_score(metrics: dict) -> float:
    return float((metrics["coarse_dice"] + 1.0 - metrics["coarse_nhd"]) / 2.0)


def metric_summary(rows: list[dict], key: str) -> dict[str, float]:
    values = np.asarray([float(row[key]) for row in rows], dtype=float)
    return {
        "mean": float(values.mean()),
        "sample_standard_deviation": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
    }


def prepare_coarse_prediction(semantic: torch.Tensor, labelmap: LabelMap) -> dict:
    values = semantic.float().clamp_min(0)
    fine_probability = (values / values.sum(0, keepdim=True).clamp_min(1e-8)).cpu().numpy()
    fine_base = fine_probability.argmax(0).astype(np.uint8)
    base = labelmap.fine_to_coarse[fine_base].astype(np.uint8)

    coarse_probability = np.zeros((len(labelmap.coarse_names), *base.shape), dtype=np.float32)
    for fine_id, coarse_id in enumerate(labelmap.fine_to_coarse):
        coarse_probability[int(coarse_id)] += fine_probability[fine_id]
    alternatives = coarse_probability.copy()
    np.put_along_axis(alternatives, base[None], -1.0, axis=0)
    second = alternatives.argmax(0).astype(np.uint8)

    components: list[dict] = []
    for class_id in range(1, len(labelmap.coarse_names)):
        labels, count = connected_components(base == class_id)
        if not count:
            continue
        sizes = np.bincount(labels.ravel(), minlength=count + 1)
        for component_id in range(1, count + 1):
            flat_indices = np.flatnonzero(labels.ravel() == component_id)
            components.append({
                "class_id": class_id,
                "class_name": labelmap.coarse_names[class_id],
                "weight": float(labelmap.coarse_weights[class_id]),
                "area": int(sizes[component_id]),
                "mean_confidence": float(
                    coarse_probability[class_id].ravel()[flat_indices].mean()
                ),
                "flat_indices": flat_indices,
            })
    return {"base": base, "second": second, "components": components}


def apply_rule(prepared: dict, minimum_area: int, minimum_confidence: float) -> np.ndarray:
    output = prepared["base"].copy()
    if minimum_area <= 0:
        return output
    flat_output = output.ravel()
    flat_second = prepared["second"].ravel()
    for component in prepared["components"]:
        class_area_threshold = max(1, math.ceil(minimum_area / component["weight"]))
        if (component["area"] < class_area_threshold
                and component["mean_confidence"] < minimum_confidence):
            indices = component["flat_indices"]
            flat_output[indices] = flat_second[indices]
    return output


def coarse_metrics(predictions: list[np.ndarray], targets: list[np.ndarray],
                   cases: list[str], labelmap: LabelMap) -> dict:
    details = segmentation_metrics(
        predictions, targets, cases, labelmap.coarse_weights, labelmap.coarse_names,
    )
    result = {
        "coarse_dice": details["dice"],
        "coarse_nhd": details["nhd"],
        "coarse_pixel_accuracy": float(np.mean([
            (prediction == target).mean() for prediction, target in zip(predictions, targets)
        ])),
    }
    result["task2_score"] = task2_score(result)
    return result


@torch.no_grad()
def infer_split(model, config: dict, data_root: Path, labelmap: LabelMap, fold: int,
                split: str, device: torch.device) -> tuple[list[dict], list[np.ndarray], list[str], list[str]]:
    dataset = TigerMultitaskDataset(
        records_for_fold(data_root, fold, split), labelmap,
        config["height"], config["width"], training=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate)
    prepared, targets, cases, names = [], [], [], []
    for batch in loader:
        output = model(pixel_values=batch["image"].to(device))
        semantic = torch.nn.functional.interpolate(
            semantic_logits(output), size=batch["fine"].shape[-2:],
            mode="bilinear", align_corners=False,
        )[0]
        prepared.append(prepare_coarse_prediction(semantic, labelmap))
        fine_target = batch["fine"].numpy()[0]
        targets.append(labelmap.fine_to_coarse[fine_target].astype(np.uint8))
        cases.append(batch["case_id"][0])
        names.append(batch["name"][0])
    return prepared, targets, cases, names


def tune_rule(prepared: list[dict], targets: list[np.ndarray], cases: list[str],
              labelmap: LabelMap, minimum_gain: float) -> tuple[dict, list[dict]]:
    candidates = [{"minimum_area": 0, "minimum_confidence": 0.0}]
    candidates.extend(
        {"minimum_area": area, "minimum_confidence": confidence}
        for area in (16, 64, 256)
        for confidence in (0.45, 0.60, 0.75, 1.01)
    )
    search = []
    for candidate in candidates:
        predictions = [apply_rule(item, **candidate) for item in prepared]
        search.append({**candidate, **coarse_metrics(predictions, targets, cases, labelmap)})
    baseline = search[0]
    best = max(search, key=lambda row: (
        row["task2_score"], -row["minimum_area"], -row["minimum_confidence"],
    ))
    if best["task2_score"] < baseline["task2_score"] + minimum_gain:
        best = baseline
    selected = {
        "minimum_area": int(best["minimum_area"]),
        "minimum_confidence": float(best["minimum_confidence"]),
        "validation_gain": float(best["task2_score"] - baseline["task2_score"]),
    }
    return selected, search


def run_fold(args: argparse.Namespace, config: dict, labelmap: LabelMap,
             fold: int, device: torch.device) -> dict:
    checkpoint_path = args.model_dir / f"fold_{fold}" / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()

    validation_prepared, validation_targets, validation_cases, _ = infer_split(
        model, config, args.data_root, labelmap, fold, "validation", device,
    )
    selected, search = tune_rule(
        validation_prepared, validation_targets, validation_cases,
        labelmap, args.minimum_validation_gain,
    )
    fold_dir = args.output_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    (fold_dir / "validation_search.json").write_text(json.dumps(search, indent=2) + "\n")
    (fold_dir / "calibration.json").write_text(json.dumps(selected, indent=2) + "\n")

    test_prepared, test_targets, test_cases, test_names = infer_split(
        model, config, args.data_root, labelmap, fold, "test", device,
    )
    baseline_predictions = [item["base"] for item in test_prepared]
    tuned_predictions = [
        apply_rule(item, selected["minimum_area"], selected["minimum_confidence"])
        for item in test_prepared
    ]
    baseline_metrics = coarse_metrics(
        baseline_predictions, test_targets, test_cases, labelmap,
    )
    tuned_metrics = coarse_metrics(tuned_predictions, test_targets, test_cases, labelmap)

    prediction_dir = fold_dir / "test_predictions_coarse_ids"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for name, prediction in zip(test_names, tuned_predictions):
        Image.fromarray(prediction, mode="L").save(prediction_dir / name)
    result = {
        "fold": fold,
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_epoch": checkpoint.get("epoch"),
        "calibration_split": "validation",
        "calibration_cases": sorted(set(validation_cases)),
        "test_cases": sorted(set(test_cases)),
        "selected_rule": selected,
        "baseline": baseline_metrics,
        "tuned": tuned_metrics,
        "test_gain": tuned_metrics["task2_score"] - baseline_metrics["task2_score"],
    }
    (fold_dir / "test_quantitative.json").write_text(json.dumps(result, indent=2) + "\n")
    del model
    torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    if args.output_dir.resolve() == args.model_dir.resolve():
        raise ValueError("Tuned output directory must differ from source model directory")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty tuned version: {args.output_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("Task-2 tuning must run on a CUDA node")
    config = yaml.safe_load(args.config.read_text())
    if config.get("experiment_id") not in {"mask2former_swin_tiny", "mask2former_swin_small"}:
        raise ValueError("This v1 tuner supports Mask2Former Swin-Tiny/Small only")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    labelmap = LabelMap.load(args.data_root / "labelmap.csv")
    metadata = {
        "version": args.output_dir.name,
        "source_experiment": config["experiment_id"],
        "source_model_dir": str(args.model_dir),
        "tuning_scope": "Task 2 only",
        "baseline_definition": "fine argmax mapped through labelmap fine_to_coarse",
        "test_ground_truth_used_for_tuning": False,
        "candidate_minimum_areas": [0, 16, 64, 256],
        "candidate_minimum_confidences": [0.45, 0.60, 0.75, 1.01],
        "minimum_validation_gain": args.minimum_validation_gain,
    }
    (args.output_dir / "version.json").write_text(json.dumps(metadata, indent=2) + "\n")
    device = torch.device("cuda")
    folds = [run_fold(args, config, labelmap, fold, device) for fold in range(5)]
    keys = ("coarse_dice", "coarse_nhd", "coarse_pixel_accuracy", "task2_score")
    summary = {
        stage: {key: metric_summary([row[stage] for row in folds], key) for key in keys}
        for stage in ("baseline", "tuned")
    }
    summary["task2_score_gain"] = metric_summary(folds, "test_gain")
    folds_improved = sum(row["test_gain"] > 0 for row in folds)
    output = {
        **metadata,
        "completed_test_folds": len(folds),
        "folds": folds,
        "summary": summary,
        "acceptance": {
            "mean_task2_score_improved": (
                summary["tuned"]["task2_score"]["mean"]
                > summary["baseline"]["task2_score"]["mean"]
            ),
            "folds_improved": folds_improved,
            "passes_four_of_five_rule": folds_improved >= 4,
        },
    }
    (args.output_dir / "test_quantitative.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
