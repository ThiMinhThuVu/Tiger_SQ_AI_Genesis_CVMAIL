#!/usr/bin/env python3
"""Run weighted TIGER test inference for DINOv2 or MedSAM checkpoints."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tiger_models.data import LabelMap, TigerMultitaskDataset, records_for_fold
from tiger_models.metrics import segmentation_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=("dinov2", "medsam"), required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=ROOT / "data")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allow-incomplete", action="store_true")
    return p.parse_args()


def build_model(kind: str, config: dict):
    if kind == "dinov2":
        from scripts.train_dinov2 import DINOv2Segmenter
        return DINOv2Segmenter(
            config["pretrained_checkpoint"], config["num_labels"],
            config["height"], config["width"], config["freeze_encoder"],
        )
    sys.path.insert(0, str(ROOT / "third_party/MedSAM"))
    from scripts.train_medsam import MedSAMSegmenter
    return MedSAMSegmenter(
        config["checkpoint"], config["num_labels"],
        config["height"], config["width"], config["freeze_encoder"],
    )


@torch.no_grad()
def infer_fold(kind: str, config: dict, model_dir: Path, fold: int,
               data_root: Path, device: torch.device) -> dict | None:
    checkpoint_path = model_dir / f"fold_{fold}" / "best.pt"
    if not checkpoint_path.is_file():
        return None
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(kind, config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    labelmap = LabelMap.load(data_root / "labelmap.csv")
    dataset = TigerMultitaskDataset(
        records_for_fold(data_root, fold, "test"), labelmap,
        config["height"], config["width"], training=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    predictions, targets, cases = [], [], []
    pixel_correct = pixel_total = 0
    for batch in loader:
        logits = model(batch["image"].to(device))
        prediction = logits.argmax(1).cpu().numpy()[0]
        target = batch["fine"].numpy()[0]
        predictions.append(prediction)
        targets.append(target)
        cases.append(batch["case_id"][0])
        pixel_correct += int((prediction == target).sum())
        pixel_total += target.size
    fine = segmentation_metrics(
        predictions, targets, cases, labelmap.fine_weights, labelmap.fine_names,
    )
    coarse_predictions = [labelmap.fine_to_coarse[item] for item in predictions]
    coarse_targets = [labelmap.fine_to_coarse[item] for item in targets]
    coarse = segmentation_metrics(
        coarse_predictions, coarse_targets, cases,
        labelmap.coarse_weights, labelmap.coarse_names,
    )
    return {
        "fold": fold,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_validation_metrics": checkpoint.get("metrics"),
        "fine_dice": fine["dice"],
        "fine_nhd": fine["nhd"],
        "coarse_dice": coarse["dice"],
        "coarse_nhd": coarse["nhd"],
        "fine_pixel_accuracy": pixel_correct / pixel_total,
        "selection_score": None,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Test inference must run on a CUDA node")
    config = yaml.safe_load(args.config.read_text())
    device = torch.device("cuda")
    results = []
    for fold in range(5):
        result = infer_fold(args.model, config, args.model_dir, fold, args.data_root, device)
        if result is not None:
            results.append(result)
    if len(results) < 5 and not args.allow_incomplete:
        raise RuntimeError(f"Only {len(results)}/5 test checkpoints available")
    metric_names = ("fine_dice", "fine_nhd", "coarse_dice", "coarse_nhd", "fine_pixel_accuracy")
    summary = {
        name: {
            "mean": float(np.mean([row[name] for row in results])),
            "sample_standard_deviation": float(np.std([row[name] for row in results], ddof=1))
            if len(results) > 1 else 0.0,
        }
        for name in metric_names
    }
    output = {
        "experiment": config["experiment_id"],
        "model_family": args.model,
        "completed_test_folds": len(results),
        "selection_score": None,
        "folds": results,
        "summary": summary,
        "note": (
            "Segmentation-only test evaluation with weighted TIGER fine/coarse metrics. "
            "This runner has no visibility head; selection_score is unavailable."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
