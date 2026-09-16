#!/usr/bin/env python3
"""Run quantitative test inference for Mask2Former fold checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from scripts.train_mask2former import build_model, collate, semantic_logits
from tiger_models.data import LabelMap, TigerMultitaskDataset, records_for_fold
from tiger_models.metrics import segmentation_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=ROOT / "data")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allow-incomplete", action="store_true")
    return p.parse_args()


@torch.no_grad()
def infer_fold(config, model_dir, fold, data_root, device):
    checkpoint_path = model_dir / f"fold_{fold}" / "best.pt"
    if not checkpoint_path.is_file():
        return None
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    labelmap = LabelMap.load(data_root / "labelmap.csv")
    dataset = TigerMultitaskDataset(records_for_fold(data_root, fold, "test"), labelmap,
                                    config["height"], config["width"], training=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate)
    predictions, targets, cases = [], [], []
    pixel_correct = pixel_total = 0
    for batch in loader:
        output = model(pixel_values=batch["image"].to(device))
        logits = torch.nn.functional.interpolate(
            semantic_logits(output), size=batch["fine"].shape[-2:], mode="bilinear", align_corners=False
        )
        prediction = logits.argmax(1).cpu().numpy()[0]
        target = batch["fine"].numpy()[0]
        predictions.append(prediction); targets.append(target); cases.append(batch["case_id"][0])
        pixel_correct += int((prediction == target).sum()); pixel_total += target.size
    fine = segmentation_metrics(predictions, targets, cases, labelmap.fine_weights, labelmap.fine_names)
    coarse_predictions = [labelmap.fine_to_coarse[x] for x in predictions]
    coarse_targets = [labelmap.fine_to_coarse[x] for x in targets]
    coarse = segmentation_metrics(coarse_predictions, coarse_targets, cases,
                                   labelmap.coarse_weights, labelmap.coarse_names)
    return {
        "fold": fold, "checkpoint": str(checkpoint_path), "fine_dice": fine["dice"],
        "fine_nhd": fine["nhd"], "coarse_dice": coarse["dice"], "coarse_nhd": coarse["nhd"],
        "fine_pixel_accuracy": pixel_correct / pixel_total,
        "selection_score": None,
    }


def main():
    args = parse_args(); config = yaml.safe_load(args.config.read_text())
    if not torch.cuda.is_available():
        raise RuntimeError("Test inference must run on a CUDA node")
    results = []
    device = torch.device("cuda")
    for fold in range(5):
        result = infer_fold(config, args.model_dir, fold, args.data_root, device)
        if result is not None: results.append(result)
    if len(results) < 5 and not args.allow_incomplete:
        raise RuntimeError(f"Only {len(results)}/5 test checkpoints available; pass --allow-incomplete")
    metric_names = ["fine_dice", "fine_nhd", "coarse_dice", "coarse_nhd", "fine_pixel_accuracy"]
    summary = {name: {"mean": float(np.mean([r[name] for r in results])),
                      "sample_standard_deviation": float(np.std([r[name] for r in results], ddof=1)) if len(results)>1 else 0.0}
               for name in metric_names}
    output = {"experiment": config["experiment_id"], "completed_test_folds": len(results),
              "selection_score": None, "folds": results, "summary": summary,
              "note": "Mask2Former runner has no visibility head; official selection_score is unavailable."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
