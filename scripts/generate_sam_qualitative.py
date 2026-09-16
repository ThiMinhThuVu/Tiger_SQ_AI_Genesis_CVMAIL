#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tiger_models.data import STATIONS, LabelMap, records_for_fold
from tiger_models.metrics import segmentation_metrics
from tiger_models.runtime import atomic_json


ERROR_COLORS = np.asarray([
    [20, 20, 20], [255, 210, 0], [230, 30, 30], [30, 90, 230], [220, 30, 220]
], dtype=np.uint8)


def boundary(labels: np.ndarray) -> np.ndarray:
    result = np.zeros(labels.shape, dtype=bool)
    result[1:, :] |= labels[1:, :] != labels[:-1, :]
    result[:-1, :] |= labels[:-1, :] != labels[1:, :]
    result[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    result[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    return result


def error_map(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    code = np.zeros(target.shape, dtype=np.uint8)
    different = target != prediction
    code[different & (target != 0) & (prediction != 0)] = 1
    code[(target == 0) & (prediction != 0)] = 2
    code[(target != 0) & (prediction == 0)] = 3
    code[boundary(target) ^ boundary(prediction)] = 4
    return ERROR_COLORS[code]


def load_visibility(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        f"{row['case_id']}.png": np.asarray([float(row[s]) for s in STATIONS])
        for row in rows
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labelmap = LabelMap.load(args.data_root / "labelmap.csv")
    records = records_for_fold(args.data_root, args.fold, args.split)
    visibility_prediction = load_visibility(args.prediction_dir / "task3.csv")
    samples = []
    class_frame_count = np.zeros(31, dtype=int)
    decoded_targets: dict[str, np.ndarray] = {}
    decoded_predictions: dict[str, np.ndarray] = {}
    for record in records:
        target = labelmap.decode_fine(np.asarray(Image.open(record.fine_mask).convert("RGB")))
        prediction = labelmap.decode_fine(np.asarray(Image.open(args.prediction_dir / "task1" / record.name).convert("RGB")))
        decoded_targets[record.name] = target
        decoded_predictions[record.name] = prediction
        class_frame_count[np.unique(target)] += 1
        size = (896, 512)
        target_small = np.asarray(Image.fromarray(target).resize(size, Image.Resampling.NEAREST))
        prediction_small = np.asarray(Image.fromarray(prediction).resize(size, Image.Resampling.NEAREST))
        score = segmentation_metrics(
            [prediction_small], [target_small], [record.case_id],
            labelmap.fine_weights, labelmap.fine_names,
        )
        visibility_errors = int(
            ((visibility_prediction[record.name] >= 0.5).astype(int) != record.visibility).sum()
        )
        samples.append({
            "name": record.name, "dice": score["dice"], "nhd": score["nhd"],
            "weight3": bool(np.isin(np.unique(target), np.where(labelmap.fine_weights == 3)[0]).any()),
            "visibility_labels": int(record.visibility.sum()),
            "visibility_errors": visibility_errors,
        })
    ordered = sorted(samples, key=lambda row: row["dice"])
    rare_ids = set(np.where((class_frame_count > 0) & (class_frame_count <= 2))[0].tolist())
    rare_samples = [
        row for row in samples if rare_ids & set(np.unique(decoded_targets[row["name"]]).tolist())
    ]
    selections = {
        "best": max(samples, key=lambda row: row["dice"]),
        "median": ordered[len(ordered) // 2],
        "worst": min(samples, key=lambda row: row["dice"]),
        "highest_nhd": max(samples, key=lambda row: row["nhd"]),
        "clinical_weight_3": min((r for r in samples if r["weight3"]), key=lambda row: row["dice"]),
        "rare_class": min(rare_samples, key=lambda row: row["dice"]) if rare_samples else min(samples, key=lambda row: row["dice"]),
        "many_visibility_labels": max(samples, key=lambda row: row["visibility_labels"]),
        "many_task3_errors": max(samples, key=lambda row: row["visibility_errors"]),
    }

    record_by_name = {record.name: record for record in records}
    for category, row in selections.items():
        record = record_by_name[row["name"]]
        image = np.asarray(Image.open(record.image).convert("RGB"))
        target = decoded_targets[record.name]
        prediction = decoded_predictions[record.name]
        target_coarse = labelmap.fine_to_coarse[target]
        prediction_coarse = labelmap.fine_to_coarse[prediction]
        entropy = np.load(args.prediction_dir / "entropy" / f"{Path(record.name).stem}.npy")
        probability = visibility_prediction[record.name]

        fig, axes = plt.subplots(2, 5, figsize=(24, 10))
        panels = axes.ravel()
        panels[0].imshow(image); panels[0].set_title("A. Original")
        panels[1].imshow(labelmap.encode_fine(target)); panels[1].set_title("B. Fine ground truth")
        panels[2].imshow(labelmap.encode_fine(prediction)); panels[2].set_title("C. Fine prediction")
        panels[3].imshow(error_map(target, prediction)); panels[3].set_title("D. Fine semantic error")
        panels[4].imshow(image)
        panels[4].contour(boundary(target), levels=[0.5], colors="lime", linewidths=1.0, linestyles="solid")
        panels[4].contour(boundary(prediction), levels=[0.5], colors="red", linewidths=1.0, linestyles="dashed")
        panels[4].set_title("E. Boundaries: GT solid / pred dashed")
        panels[5].imshow(labelmap.encode_coarse(target_coarse)); panels[5].set_title("F. Coarse ground truth")
        panels[6].imshow(labelmap.encode_coarse(prediction_coarse)); panels[6].set_title("G. Coarse prediction")
        panels[7].imshow(error_map(target_coarse, prediction_coarse)); panels[7].set_title("H. Coarse error")
        panels[8].imshow(entropy, cmap="magma", vmin=0, vmax=1); panels[8].set_title("I. Normalized prediction entropy")
        colors = ["tab:green" if truth else "tab:blue" for truth in record.visibility]
        panels[9].barh(np.arange(14), probability, color=colors)
        panels[9].axvline(0.5, color="black", linestyle="--")
        panels[9].set_yticks(np.arange(14), STATIONS); panels[9].invert_yaxis(); panels[9].set_xlim(0, 1)
        panels[9].set_title("J. Task 3 probability (green=GT positive)")
        for panel in panels[:9]:
            panel.axis("off")
        fig.suptitle(f"{category}: {record.name} | Dice={row['dice']:.4f}, nHD={row['nhd']:.4f}")
        fig.tight_layout()
        fig.savefig(args.output_dir / f"{category}_{Path(record.name).stem}.png", dpi=130)
        plt.close(fig)
    atomic_json(args.output_dir / "selections.json", selections)
    lines = [f"# SAM qualitative error analysis — Fold {args.fold} ({args.split})", ""]
    for category, row in selections.items():
        figure = f"{category}_{Path(row['name']).stem}.png"
        lines.extend([
            f"## {category.replace('_', ' ').title()}", "",
            f"Frame `{row['name']}`; fine Dice {row['dice']:.4f}; fine nHD {row['nhd']:.4f}; "
            f"visibility errors {row['visibility_errors']}.", "", f"![{category}]({figure})", "",
        ])
    (args.output_dir / "SAM_ERROR_ANALYSIS.md").write_text("\n".join(lines))
    print(json.dumps(selections, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
