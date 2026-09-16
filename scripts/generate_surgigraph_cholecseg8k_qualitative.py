#!/usr/bin/env python3
"""Render best/median/worst test frames as Input | GT | Prediction panels."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_surgigraph_cholecseg8k import CholecDataset, build_model, load_pairs


PALETTE = np.asarray([
    [0, 0, 0], [230, 159, 0], [0, 158, 115], [86, 180, 233],
    [204, 121, 167], [213, 94, 0], [0, 114, 178], [240, 228, 66],
    [128, 0, 128], [0, 200, 200], [220, 20, 60], [120, 120, 255],
    [160, 100, 40],
], dtype=np.uint8)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=4)
    return parser.parse_args()


def main():
    args = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    config = yaml.safe_load(args.config.read_text())
    experiment = args.output_root / config["experiment_id"]
    test_result = json.loads((experiment / f"test_fold{args.fold}.json").read_text())
    frame_metrics = test_result["fine"]["per_image"]
    pairs = load_pairs(args.data_root, args.fold, "test")
    if len(frame_metrics) != len(pairs):
        raise RuntimeError("Test metric order does not match the frozen test split")

    ranked = sorted(range(len(pairs)), key=lambda index: frame_metrics[index]["dice"])
    selected = {
        "best": ranked[-1],
        "mid": ranked[len(ranked) // 2],
        "worst": ranked[0],
    }

    device = torch.device("cuda")
    model = build_model(config).to(device)
    checkpoint = experiment / f"fold_{args.fold}" / "best.pt"
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"])
    model.eval()
    dataset = CholecDataset(args.data_root, pairs, config["height"], config["width"], False)

    output = experiment / f"qualitative_fold{args.fold}"
    output.mkdir(parents=True, exist_ok=True)
    manifest = {}
    with torch.no_grad():
        for category, index in selected.items():
            sample = dataset[index]
            result = model(sample["image"].unsqueeze(0).to(device))
            probability = F.interpolate(
                result["fine_probability"], sample["fine"].shape,
                mode="bilinear", align_corners=False,
            )
            prediction = probability.argmax(1).cpu().numpy()[0]
            target = sample["fine"].numpy()
            image = np.asarray(
                Image.open(args.data_root / pairs[index]["image"]).convert("RGB").resize(
                    (config["width"], config["height"]), Image.Resampling.BILINEAR
                )
            )
            metric = frame_metrics[index]
            figure, axes = plt.subplots(1, 3, figsize=(16, 5.5))
            for axis, panel, title in zip(
                axes, (image, PALETTE[target], PALETTE[prediction]),
                ("Input", "Ground Truth", "Prediction"),
            ):
                axis.imshow(panel)
                axis.set_title(title, fontsize=15)
                axis.axis("off")
            figure.suptitle(
                f"Fold {args.fold} — {category.upper()} | Dice={metric['dice']:.4f}, "
                f"nHD={metric['nhd']:.4f}\n{pairs[index]['image']}", fontsize=13
            )
            figure.tight_layout()
            path = output / f"{category}_input_gt_prediction.png"
            figure.savefig(path, dpi=170, bbox_inches="tight")
            plt.close(figure)
            manifest[category] = {
                "index": index,
                "image": pairs[index]["image"],
                "mask": pairs[index]["mask"],
                "dice": metric["dice"],
                "nhd": metric["nhd"],
                "figure": path.name,
            }
    (output / "selections.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
