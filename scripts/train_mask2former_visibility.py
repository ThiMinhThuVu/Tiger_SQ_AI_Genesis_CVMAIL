#!/usr/bin/env python3
"""Train a 14-label visibility probe on frozen Mask2Former backbone features.

The baseline uses threshold 0.5. Tuned v1 chooses one threshold per station
from pooled out-of-fold validation predictions and applies the frozen thresholds
to local test predictions. Test targets are never used for model or threshold
selection.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_mask2former import build_model as build_baseline_model
from scripts.train_mask2former_improved import build_improved
from tiger_models.data import LabelMap, STATIONS, TigerMultitaskDataset, records_for_fold, visibility_pos_weight
from tiger_models.metrics import selection_score, visibility_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--segmentation-results", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def seed_everything(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def collate(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "visibility": torch.stack([item["visibility"] for item in batch]),
        "name": [str(item["name"]) for item in batch],
    }


def build_feature_model(config: dict, labelmap: LabelMap) -> nn.Module:
    if config.get("model_type") == "mask2former_improved":
        return build_improved(config, labelmap)
    return build_baseline_model(config)


@torch.no_grad()
def extract_features(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    model.eval()
    features, targets, names = [], [], []
    for batch in loader:
        images = batch["image"].to(device)
        if hasattr(model, "segmenter"):
            output = model(images)
            encoded = output["feature"]
        else:
            output = model(pixel_values=images, output_hidden_states=True)
            encoded = output.encoder_last_hidden_state
        if encoded is None or encoded.ndim != 4:
            raise RuntimeError("Expected Mask2Former encoder_last_hidden_state with shape [B,C,H,W]")
        features.append(encoded.float().mean(dim=(-2, -1)).cpu())
        targets.append(batch["visibility"].float())
        names.extend(batch["name"])
    return torch.cat(features), torch.cat(targets), names


class VisibilityHead(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(channels), nn.Linear(channels, 256), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(256, len(STATIONS)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


def train_head(
    train_x: torch.Tensor, train_y: torch.Tensor, val_x: torch.Tensor, val_y: torch.Tensor,
    pos_weight: torch.Tensor, device: torch.device, ns: argparse.Namespace,
) -> tuple[VisibilityHead, list[dict[str, float]]]:
    head = VisibilityHead(train_x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=ns.learning_rate, weight_decay=ns.weight_decay)
    train_x, train_y = train_x.to(device), train_y.to(device)
    val_x, val_y = val_x.to(device), val_y.to(device)
    pos_weight = pos_weight.to(device)
    best_loss, best_state, stale, history = float("inf"), None, 0, []
    for epoch in range(ns.epochs):
        head.train()
        permutation = torch.randperm(len(train_x), device=device)
        losses = []
        for start in range(0, len(train_x), 16):
            index = permutation[start:start + 16]
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(head(train_x[index]), train_y[index], pos_weight=pos_weight)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        head.eval()
        with torch.no_grad():
            val_loss = float(F.binary_cross_entropy_with_logits(head(val_x), val_y, pos_weight=pos_weight).cpu())
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss})
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= ns.patience:
                break
    if best_state is None:
        raise RuntimeError("Visibility head training did not produce a checkpoint")
    head.load_state_dict(best_state)
    return head, history


def probabilities(head: VisibilityHead, features: torch.Tensor, device: torch.device) -> np.ndarray:
    head.eval()
    with torch.no_grad():
        return head(features.to(device)).sigmoid().cpu().numpy()


def tune_thresholds(target: np.ndarray, probability: np.ndarray) -> np.ndarray:
    thresholds = np.full(target.shape[1], 0.5, dtype=np.float64)
    grid = np.arange(0.05, 0.951, 0.025)
    for column in range(target.shape[1]):
        truth = target[:, column].astype(bool)
        if truth.sum() == 0:
            continue
        best = (-1.0, -1.0, 0.5)
        for threshold in grid:
            prediction = probability[:, column] >= threshold
            tp = np.logical_and(prediction, truth).sum()
            fp = np.logical_and(prediction, ~truth).sum()
            fn = np.logical_and(~prediction, truth).sum()
            f1 = 2 * tp / max(2 * tp + fp + fn, 1)
            candidate = (float(f1), -abs(float(threshold) - 0.5), float(threshold))
            if candidate > best:
                best = candidate
        thresholds[column] = best[2]
    return thresholds


def thresholded_metrics(target: np.ndarray, probability: np.ndarray, thresholds: np.ndarray) -> dict[str, object]:
    # visibility_metrics uses 0.5; map frozen per-class decisions to exact binary probabilities for F1,
    # while retaining raw probabilities for the threshold-independent AUROC.
    raw = visibility_metrics(target, probability, STATIONS)
    decisions = (probability >= thresholds[None, :]).astype(np.float64)
    calibrated = visibility_metrics(target, decisions, STATIONS)
    return {"macro_f1": calibrated["macro_f1"], "macro_auroc": raw["macro_auroc"],
            "accuracy": float((decisions == target).mean()), "per_station": calibrated["per_station"]}


def aggregate(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=1))}


def main() -> int:
    ns = parse_args()
    if not torch.cuda.is_available() and not ns.allow_cpu:
        raise RuntimeError("CUDA unavailable; use --allow-cpu only for engineering checks")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = yaml.safe_load(ns.config.read_text())
    labelmap = LabelMap.load(ns.data_root / "labelmap.csv")
    if ns.split_manifest:
        from full_version.task2_fine_31cls.scripts.data_full import (
            build_manifest, load_manifest, visibility_records_for_manifest,
        )
        manifest = load_manifest(ns.split_manifest)
        if manifest.get("version") != "full40_case_folds_v1":
            raise ValueError(f"Unsupported split manifest: {manifest.get('version')}")
        current = build_manifest(ns.data_root, int(config["split_seed"]))
        if current["snapshot"] != manifest.get("snapshot"):
            raise ValueError("Current dataset snapshot does not match the split manifest")
        records_provider = lambda fold, split: visibility_records_for_manifest(
            ns.data_root, manifest, fold, split
        )
    else:
        records_provider = lambda fold, split: records_for_fold(ns.data_root, fold, split)
    fold_records, oof_target, oof_probability = [], [], []
    ns.output_root.mkdir(parents=True, exist_ok=True)

    for fold in range(5):
        seed_everything(int(config["seed"]) + 300 + fold)
        records = {split: records_provider(fold, split) for split in ("train", "validation", "test")}
        if any(not rows for rows in records.values()):
            raise ValueError(f"Fold {fold} has an empty labeled visibility split")
        loaders = {}
        for split in records:
            dataset = TigerMultitaskDataset(records[split], labelmap, config["height"], config["width"], training=False)
            loaders[split] = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=config["num_workers"],
                                        pin_memory=device.type == "cuda", collate_fn=collate)
        model = build_feature_model(config, labelmap)
        checkpoint_path = ns.model_dir / f"fold_{fold}/best.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        model.to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        extracted = {split: extract_features(model, loaders[split], device) for split in loaders}
        del model
        torch.cuda.empty_cache()
        pos_weight = torch.from_numpy(visibility_pos_weight(records["train"]))
        head, history = train_head(extracted["train"][0], extracted["train"][1],
                                   extracted["validation"][0], extracted["validation"][1],
                                   pos_weight, device, ns)
        fold_dir = ns.output_root / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"visibility_head": head.state_dict(), "feature_channels": extracted["train"][0].shape[1],
                    "source_checkpoint": str(checkpoint_path), "history": history}, fold_dir / "best_visibility.pt")
        val_probability = probabilities(head, extracted["validation"][0], device)
        test_probability = probabilities(head, extracted["test"][0], device)
        np.savez_compressed(fold_dir / "visibility_predictions.npz",
                            validation_names=np.asarray(extracted["validation"][2]),
                            validation_target=extracted["validation"][1].numpy(),
                            validation_probability=val_probability,
                            test_names=np.asarray(extracted["test"][2]), test_target=extracted["test"][1].numpy(),
                            test_probability=test_probability)
        oof_target.append(extracted["validation"][1].numpy())
        oof_probability.append(val_probability)
        fold_records.append({"fold": fold, "test_target": extracted["test"][1].numpy(),
                             "test_probability": test_probability,
                             "test_names": extracted["test"][2]})
        print(json.dumps({"fold": fold, "epochs": len(history), "best_val_loss": min(x["val_loss"] for x in history)}), flush=True)

    thresholds = tune_thresholds(np.concatenate(oof_target), np.concatenate(oof_probability))
    segmentation_path = ns.segmentation_results or (ns.model_dir / "test_quantitative.json")
    segmentation = json.loads(segmentation_path.read_text())
    segmentation_by_fold = {row["fold"]: row for row in segmentation["folds"]}
    outputs = {}
    for version, version_thresholds in (("base", np.full(len(STATIONS), 0.5)), ("tuned_v1", thresholds)):
        rows = []
        for record in fold_records:
            visibility = thresholded_metrics(record["test_target"], record["test_probability"], version_thresholds)
            seg = segmentation_by_fold[record["fold"]]
            score = selection_score(seg["fine_dice"], seg["fine_nhd"], seg["coarse_dice"], seg["coarse_nhd"],
                                    visibility["macro_f1"], visibility["macro_auroc"])
            rows.append({"fold": record["fold"], "visibility_macro_f1": visibility["macro_f1"],
                         "visibility_macro_auroc": visibility["macro_auroc"], "visibility_accuracy": visibility["accuracy"],
                         "selection_score": score})
        summary = {metric: aggregate([row[metric] for row in rows]) for metric in
                   ("visibility_macro_f1", "visibility_macro_auroc", "visibility_accuracy", "selection_score")}
        all_test_target = np.concatenate([record["test_target"] for record in fold_records])
        all_test_probability = np.concatenate([record["test_probability"] for record in fold_records])
        overall_visibility = thresholded_metrics(all_test_target, all_test_probability, version_thresholds)
        output = {"experiment": f"{config['experiment_id']}_task3_{version}", "completed_test_folds": 5,
                  "labeled_test_frames": int(len(all_test_target)),
                  "source_segmentation": str(segmentation_path), "threshold_source": "0.5" if version == "base" else "pooled OOF validation",
                  "thresholds": dict(zip(STATIONS, version_thresholds.tolist())), "folds": rows,
                  "oof_visibility": overall_visibility, "summary": summary}
        output_dir = ns.output_root.parent / f"{ns.output_root.name}_{version}"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "test_quantitative.json").write_text(json.dumps(output, indent=2))
        outputs[version] = output
    (ns.output_root / "completion.json").write_text(json.dumps({"status": "completed", "versions": list(outputs)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
