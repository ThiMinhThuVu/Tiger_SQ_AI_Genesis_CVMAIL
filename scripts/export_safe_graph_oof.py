#!/usr/bin/env python3
"""Export validation-only OOF evidence for SAFE-Graph verifier development.

This exporter never reads the local held-out test split. It stores compact top-k
pixel evidence, optionally requested dense class-probability maps, and raw
component/GT overlap measurements; final error labels are left unadjudicated for
a separately frozen taxonomy.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy import ndimage
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_mask2former import collate  # noqa: E402
from scripts.train_mask2former_improved import build_improved  # noqa: E402
from tiger_models.data import (  # noqa: E402
    STATIONS, LabelMap, Record, TigerMultitaskDataset, records_for_fold, sha256,
)
from tiger_models.improved_mask2former import soft_presence_modulation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "configs/mask2former_swin_small_clinical_hier_presence_v2.yaml",
    )
    parser.add_argument(
        "--model-dir", type=Path,
        default=ROOT / "artifacts/mask2former_improved/mask2former_swin_small_clinical_hier_presence_v2",
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", choices=range(5), default=list(range(5)))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--explicit-probability-class-ids", type=int, nargs="*", choices=range(31),
        default=[], help="Save dense probability maps and component means for these classes",
    )
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument(
        "--reference-oof-root", type=Path, default=None,
        help="Reuse the exact validation frame list/station targets from a frozen OOF export",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def top_classes(mean_probability: np.ndarray, k: int = 3) -> tuple[list[int], list[float]]:
    order = np.argsort(mean_probability)[::-1][:k]
    return order.astype(int).tolist(), mean_probability[order].astype(float).tolist()


def normalized_tool_distance(tool_mask: np.ndarray, component: np.ndarray) -> float | None:
    if not tool_mask.any():
        return None
    diagonal = math.hypot(*component.shape)
    distance = ndimage.distance_transform_edt(~tool_mask)
    return float(distance[component].min(initial=diagonal) / max(diagonal, 1.0))


def validation_records_from_reference(
    data_root: Path, fold: int, reference_oof_root: Path,
) -> list[Record]:
    fold_root = reference_oof_root / f"fold_{fold}"
    frame_rows = list(csv.DictReader((fold_root / "frame_index.csv").open(newline="")))
    records = []
    for row in frame_rows:
        if int(row["fold"]) != fold or row["split"] != "validation_oof":
            raise ValueError(f"Invalid reference frame row for fold {fold}: {row}")
        array_path = fold_root / row["array_file"]
        with np.load(array_path) as arrays:
            station_target = arrays["station_target"].astype(np.float32)
        if station_target.shape != (len(STATIONS),):
            raise ValueError(f"Invalid station target shape for {row['frame']}")
        image = data_root / "images" / row["frame"]
        fine_mask = data_root / "masks_fine" / row["frame"]
        if not image.is_file() or not fine_mask.is_file():
            raise FileNotFoundError(f"Missing image/mask for reference frame {row['frame']}")
        records.append(Record(
            row["frame"], row["case_id"], image, fine_mask, station_target,
        ))
    if len(records) != 14 or len({record.case_id for record in records}) != 1:
        raise ValueError(f"Reference fold {fold} is not an exact 14-frame/1-case validation split")
    return records


def predicted_component_rows(
    prediction: np.ndarray,
    target: np.ndarray,
    probability: np.ndarray,
    entropy: np.ndarray,
    fold: int,
    case_id: str,
    frame: str,
    station_target: np.ndarray,
    class_names: tuple[str, ...],
    explicit_probability_class_ids: tuple[int, ...] = (),
) -> list[dict]:
    rows: list[dict] = []
    component_index = 0
    top_values = np.partition(probability, -2, axis=0)[-2:]
    margin = top_values[-1] - top_values[-2]
    predicted_tool = prediction == 1
    target_tool = target == 1
    station_names = "|".join(station for station, flag in zip(STATIONS, station_target) if bool(flag))
    for class_id in range(1, len(class_names)):
        labels, count = ndimage.label(prediction == class_id, structure=np.ones((3, 3), dtype=np.uint8))
        for local_id in range(1, count + 1):
            component = labels == local_id
            ys, xs = np.nonzero(component)
            area = int(component.sum())
            target_counts = np.bincount(target[component], minlength=len(class_names))
            dominant_gt = int(target_counts.argmax())
            same_fraction = float(target_counts[class_id] / area)
            dominant_fraction = float(target_counts[dominant_gt] / area)
            class_mean = probability[:, component].mean(1)
            alternative_ids, alternative_probabilities = top_classes(class_mean, 3)
            explicit_probabilities = [
                float(class_mean[class_id]) for class_id in explicit_probability_class_ids
            ]
            component_index += 1
            rows.append({
                "fold": fold,
                "split": "validation_oof",
                "case_id": case_id,
                "frame": frame,
                "component_id": component_index,
                "predicted_class_id": class_id,
                "predicted_class_name": class_names[class_id],
                "area_pixels": area,
                "area_fraction": area / prediction.size,
                "bbox_x0": int(xs.min()),
                "bbox_y0": int(ys.min()),
                "bbox_x1": int(xs.max()) + 1,
                "bbox_y1": int(ys.max()) + 1,
                "centroid_x_norm": float(xs.mean() / prediction.shape[1]),
                "centroid_y_norm": float(ys.mean() / prediction.shape[0]),
                "mean_predicted_class_probability": float(class_mean[class_id]),
                "mean_top1_probability": float(probability.max(0)[component].mean()),
                "mean_top1_top2_margin": float(margin[component].mean()),
                "mean_entropy": float(entropy[component].mean()),
                "mean_tool_probability": float(probability[1, component].mean()),
                "predicted_tool_distance_norm": normalized_tool_distance(predicted_tool, component),
                "gt_tool_distance_norm": normalized_tool_distance(target_tool, component),
                "dominant_gt_class_id": dominant_gt,
                "dominant_gt_class_name": class_names[dominant_gt],
                "dominant_gt_fraction": dominant_fraction,
                "same_class_gt_fraction": same_fraction,
                "alternative_class_ids": "|".join(map(str, alternative_ids)),
                "alternative_class_names": "|".join(class_names[index] for index in alternative_ids),
                "alternative_mean_probabilities": "|".join(f"{value:.8g}" for value in alternative_probabilities),
                "explicit_probability_class_ids": "|".join(
                    map(str, explicit_probability_class_ids)
                ),
                "explicit_class_mean_probabilities": "|".join(
                    f"{value:.8g}" for value in explicit_probabilities
                ),
                "visible_station_targets": station_names,
                "station_prediction_source": "unavailable_in_improved_v2",
                "error_label": "UNADJUDICATED",
            })
    return rows


def gt_component_rows(
    prediction: np.ndarray,
    target: np.ndarray,
    fold: int,
    case_id: str,
    frame: str,
    class_names: tuple[str, ...],
) -> list[dict]:
    rows: list[dict] = []
    component_index = 0
    for class_id in range(1, len(class_names)):
        labels, count = ndimage.label(target == class_id, structure=np.ones((3, 3), dtype=np.uint8))
        for local_id in range(1, count + 1):
            component = labels == local_id
            area = int(component.sum())
            prediction_counts = np.bincount(prediction[component], minlength=len(class_names))
            dominant_prediction = int(prediction_counts.argmax())
            component_index += 1
            rows.append({
                "fold": fold,
                "split": "validation_oof",
                "case_id": case_id,
                "frame": frame,
                "gt_component_id": component_index,
                "gt_class_id": class_id,
                "gt_class_name": class_names[class_id],
                "area_pixels": area,
                "same_class_prediction_coverage": float(prediction_counts[class_id] / area),
                "dominant_predicted_class_id": dominant_prediction,
                "dominant_predicted_class_name": class_names[dominant_prediction],
                "dominant_prediction_fraction": float(prediction_counts[dominant_prediction] / area),
                "complete_miss_candidate": bool(prediction_counts[class_id] == 0),
                "error_label": "UNADJUDICATED",
            })
    return rows


@torch.inference_mode()
def export_fold(
    fold: int,
    config: dict,
    model_dir: Path,
    data_root: Path,
    output_root: Path,
    labelmap: LabelMap,
    device: torch.device,
    top_k: int,
    explicit_probability_class_ids: tuple[int, ...],
    reference_oof_root: Path | None,
) -> dict:
    checkpoint_path = model_dir / f"fold_{fold}/best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    records = (
        validation_records_from_reference(data_root, fold, reference_oof_root)
        if reference_oof_root is not None
        else records_for_fold(data_root, fold, "validation")
    )
    if len(records) != 14 or len({record.case_id for record in records}) != 1:
        raise ValueError(f"Fold {fold} is not the expected 14-frame/1-case validation split")
    dataset = TigerMultitaskDataset(records, labelmap, config["height"], config["width"], False)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=2 if device.type == "cuda" else 0,
        collate_fn=collate, pin_memory=device.type == "cuda",
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_improved(config, labelmap).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    fold_dir = output_root / f"fold_{fold}"
    arrays_dir = fold_dir / "frame_arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    component_rows: list[dict] = []
    target_component_rows: list[dict] = []
    frame_rows: list[dict] = []
    for batch in loader:
        result = model(batch["image"].to(device))
        fine = F.interpolate(
            result["fine_probability"], batch["fine"].shape[-2:], mode="bilinear", align_corners=False,
        )
        if config.get("presence_head", True):
            fine = soft_presence_modulation(
                fine, result["presence_logits"],
                float(config["presence_epsilon"]), float(config["presence_gamma"]),
            )
        fine = fine / fine.sum(1, keepdim=True).clamp_min(1e-6)
        values, labels = fine.topk(top_k, dim=1)
        entropy = -(fine.clamp_min(1e-8) * fine.clamp_min(1e-8).log()).sum(1)
        prediction = labels[0, 0].byte().cpu().numpy()
        target = batch["fine"][0].byte().numpy()
        probability = fine[0].float().cpu().numpy()
        entropy_np = entropy[0].float().cpu().numpy()
        station_target = batch["visibility"][0].byte().numpy()
        presence_probability = result["presence_logits"][0].sigmoid().float().cpu().numpy()
        name, case_id = batch["name"][0], batch["case_id"][0]
        if reference_oof_root is not None:
            reference_path = (
                reference_oof_root / f"fold_{fold}/frame_arrays/{Path(name).stem}.npz"
            )
            with np.load(reference_path) as reference:
                if not np.array_equal(prediction, reference["prediction"]):
                    raise ValueError(f"Prediction differs from frozen OOF reference for {name}")
                if not np.array_equal(target, reference["target"]):
                    raise ValueError(f"Target differs from frozen OOF reference for {name}")
        array_payload = {
            "topk_labels": labels[0].byte().cpu().numpy(),
            "topk_probabilities": values[0].half().cpu().numpy(),
            "entropy": entropy[0].half().cpu().numpy(),
            "tool_probability": fine[0, 1].half().cpu().numpy(),
            "prediction": prediction,
            "target": target,
            "presence_probability": presence_probability.astype(np.float16),
            "station_target": station_target,
        }
        if explicit_probability_class_ids:
            array_payload["explicit_probability_class_ids"] = np.asarray(
                explicit_probability_class_ids, dtype=np.uint8
            )
            array_payload["explicit_class_probabilities"] = fine[
                0, list(explicit_probability_class_ids)
            ].half().cpu().numpy()
        np.savez_compressed(arrays_dir / f"{Path(name).stem}.npz", **array_payload)
        pred_rows = predicted_component_rows(
            prediction, target, probability, entropy_np, fold, case_id, name,
            station_target, labelmap.fine_names, explicit_probability_class_ids,
        )
        gt_rows = gt_component_rows(prediction, target, fold, case_id, name, labelmap.fine_names)
        component_rows.extend(pred_rows)
        target_component_rows.extend(gt_rows)
        frame_rows.append({
            "fold": fold,
            "split": "validation_oof",
            "case_id": case_id,
            "frame": name,
            "predicted_components": len(pred_rows),
            "gt_components": len(gt_rows),
            "pixel_accuracy": float((prediction == target).mean()),
            "mean_top1_probability": float(probability.max(0).mean()),
            "mean_entropy": float(entropy_np.mean()),
            "visible_station_targets": "|".join(
                station for station, flag in zip(STATIONS, station_target) if bool(flag)
            ),
            "station_prediction_source": "unavailable_in_improved_v2",
            "array_file": str(Path("frame_arrays") / f"{Path(name).stem}.npz"),
        })

    write_csv(fold_dir / "frame_index.csv", frame_rows)
    write_csv(fold_dir / "predicted_components_raw.csv", component_rows)
    write_csv(fold_dir / "gt_components_raw.csv", target_component_rows)
    manifest = {
        "fold": fold,
        "split": "validation_oof",
        "test_data_used": False,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "frames": len(frame_rows),
        "cases": sorted({row["case_id"] for row in frame_rows}),
        "predicted_components": len(component_rows),
        "gt_components": len(target_component_rows),
        "top_k": top_k,
        "explicit_probability_class_ids": list(explicit_probability_class_ids),
        "validation_record_source": (
            "frozen_oof_reference" if reference_oof_root is not None else "current_visibility_csv"
        ),
        "station_prediction_source": "unavailable_in_improved_v2",
        "error_label_status": "UNADJUDICATED",
    }
    (fold_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return manifest


def main() -> None:
    args = parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output root: {args.output_root}")
    if args.top_k < 2 or args.top_k > 5:
        raise ValueError("top-k must be between 2 and 5")
    explicit_probability_class_ids = tuple(sorted(set(args.explicit_probability_class_ids)))
    reference_manifest = None
    if args.reference_oof_root is not None:
        reference_manifest = json.loads((args.reference_oof_root / "manifest.json").read_text())
        if reference_manifest.get("split") != "out_of_fold_validation_only":
            raise ValueError("Reference OOF artifact must be validation-only")
        if reference_manifest.get("test_data_used") is not False:
            raise ValueError("Reference OOF artifact must not use test data")
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("OOF export requires CUDA; use --allow-cpu only for engineering checks")
    config = yaml.safe_load(args.config.read_text())
    labelmap = LabelMap.load(args.data_root / "labelmap.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifests = [
        export_fold(
            fold, config, args.model_dir, args.data_root, args.output_root,
            labelmap, device, args.top_k, explicit_probability_class_ids,
            args.reference_oof_root,
        )
        for fold in sorted(set(args.folds))
    ]
    all_cases = [case for item in manifests for case in item["cases"]]
    if len(manifests) == 5 and (sum(item["frames"] for item in manifests) != 70 or len(set(all_cases)) != 5):
        raise ValueError("Expected five-fold validation-only export with 70 frames from five unique cases")
    root_manifest = {
        "artifact": "SAFE-Graph compact OOF evidence corpus",
        "version": args.output_root.name,
        "experiment": config["experiment_id"],
        "split": "out_of_fold_validation_only",
        "test_data_used": False,
        "config": str(args.config),
        "config_sha256": file_sha256(args.config),
        "labelmap_sha256": sha256(args.data_root / "labelmap.csv"),
        "visibility_sha256": (
            reference_manifest["visibility_sha256"] if reference_manifest is not None
            else sha256(args.data_root / "lymph_node_station_visibility.csv")
        ),
        "folds": manifests,
        "frames": sum(item["frames"] for item in manifests),
        "cases": sorted(set(all_cases)),
        "coverage_limitation": "Validation rotation covers 5/10 project cases; held-out-test cases remain excluded.",
        "station_prediction_source": "unavailable_in_improved_v2",
        "error_label_status": "UNADJUDICATED",
        "explicit_probability_class_ids": list(explicit_probability_class_ids),
        "validation_record_source": (
            "frozen_oof_reference" if args.reference_oof_root is not None
            else "current_visibility_csv"
        ),
        "reference_oof_root": str(args.reference_oof_root) if args.reference_oof_root else None,
        "reference_oof_manifest_sha256": (
            file_sha256(args.reference_oof_root / "manifest.json")
            if args.reference_oof_root is not None else None
        ),
        "next_gate": "Freeze component matching/error taxonomy before assigning labels.",
    }
    (args.output_root / "manifest.json").write_text(json.dumps(root_manifest, indent=2) + "\n")
    print(json.dumps({
        "output_root": str(args.output_root),
        "frames": root_manifest["frames"],
        "cases": root_manifest["cases"],
        "test_data_used": False,
    }, indent=2))


if __name__ == "__main__":
    main()
