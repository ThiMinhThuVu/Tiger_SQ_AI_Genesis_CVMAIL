#!/usr/bin/env python3
"""Train Task-3 probes and produce cross-fitted validation/per-center metrics."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from full_version.task2_fine_31cls.scripts.data_full import (
    build_trainval_manifest, load_manifest, visibility_records_for_manifest,
)
from scripts.train_mask2former_visibility import (
    build_feature_model, collate, extract_features, probabilities, seed_everything,
    thresholded_metrics, train_head, tune_thresholds,
)
from tiger_models.data import LabelMap, STATIONS, TigerMultitaskDataset, visibility_pos_weight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=40)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    config = yaml.safe_load(args.config.read_text())
    manifest = load_manifest(args.manifest)
    if manifest != build_trainval_manifest(args.data_root, int(config["split_seed"])):
        raise ValueError("Manifest is not the reviewed center-stratified snapshot")
    labelmap = LabelMap.load(args.data_root / "labelmap.csv")
    visibility_csv = args.data_root / "lymph_node_station_visibility.csv"
    visibility_sha256 = hashlib.sha256(visibility_csv.read_bytes()).hexdigest()
    fold_rows = []
    for fold in range(5):
        seed_everything(int(config["seed"]) + 300 + fold)
        records = {split: visibility_records_for_manifest(args.data_root, manifest, fold, split)
                   for split in ("train", "validation")}
        extracted = {}
        model = build_feature_model(config, labelmap)
        checkpoint_path = args.model_dir / f"fold_{fold}/best.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"]); model.to(device).eval()
        for split, rows in records.items():
            dataset = TigerMultitaskDataset(rows, labelmap, config["height"], config["width"], False)
            loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=config["num_workers"],
                                pin_memory=True, collate_fn=collate)
            extracted[split] = extract_features(model, loader, device)
        del model
        torch.cuda.empty_cache()
        head, history = train_head(extracted["train"][0], extracted["train"][1],
                                   extracted["validation"][0], extracted["validation"][1],
                                   torch.from_numpy(visibility_pos_weight(records["train"])), device, args)
        probability = probabilities(head, extracted["validation"][0], device)
        fold_dir = args.output_root / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"visibility_head": head.state_dict(), "source_checkpoint": str(checkpoint_path),
                    "visibility_csv_sha256": visibility_sha256,
                    "history": history}, fold_dir / "best_visibility.pt")
        np.savez_compressed(fold_dir / "validation_predictions.npz",
                            names=np.asarray(extracted["validation"][2]),
                            target=extracted["validation"][1].numpy(), probability=probability)
        fold_rows.append({"fold": fold, "names": extracted["validation"][2],
                          "target": extracted["validation"][1].numpy(), "probability": probability})

    rows, all_target, all_probability = [], [], []
    for held_out in fold_rows:
        calibration = [row for row in fold_rows if row["fold"] != held_out["fold"]]
        thresholds = tune_thresholds(np.concatenate([row["target"] for row in calibration]),
                                     np.concatenate([row["probability"] for row in calibration]))
        metrics = thresholded_metrics(held_out["target"], held_out["probability"], thresholds)
        rows.append({"fold": held_out["fold"], "labeled_validation_frames": len(held_out["names"]),
                     "threshold_source": "other_four_validation_folds", "thresholds": thresholds.tolist(),
                     "macro_f1": metrics["macro_f1"], "macro_auroc": metrics["macro_auroc"],
                     "accuracy": metrics["accuracy"]})
        all_target.append(held_out["target"]); all_probability.append(held_out["probability"])
    per_center = {}
    for center_id in range(1, 8):
        chosen_target, chosen_probability = [], []
        for held_out, report in zip(fold_rows, rows):
            indices = [i for i, name in enumerate(held_out["names"])
                       if name.startswith(f"center_{center_id}_case_")]
            if indices:
                thresholds = np.asarray(report["thresholds"])
                chosen_target.append(held_out["target"][indices])
                chosen_probability.append((held_out["probability"][indices] >= thresholds).astype(float))
        if chosen_target:
            metric = thresholded_metrics(np.concatenate(chosen_target), np.concatenate(chosen_probability),
                                         np.full(len(STATIONS), 0.5))
            per_center[f"center_{center_id}"] = {"labeled_frames": int(sum(len(x) for x in chosen_target)),
                "macro_f1": metric["macro_f1"], "macro_auroc": metric["macro_auroc"], "accuracy": metric["accuracy"]}
    output = {"task": "task3", "split": "5_fold_oof_validation_no_local_test",
              "visibility_csv_sha256": visibility_sha256,
              "threshold_policy": "cross-fitted from the other four folds", "folds": rows,
              "per_center": per_center,
              "macro_over_centers": {key: float(np.mean([row[key] for row in per_center.values()]))
                                      for key in ("macro_f1", "macro_auroc", "accuracy")}}
    output["worst_center"] = min(per_center, key=lambda key: per_center[key]["macro_f1"])
    (args.output_root / "validation_quantitative.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["macro_over_centers"], indent=2))


if __name__ == "__main__":
    main()
