#!/usr/bin/env python3
"""Post-hoc check: was loss_sink_margin actually active in a trained checkpoint,
or did the hinge margin get trivially satisfied (term ~0)? Diagnostic only --
does not retrain anything, loads a saved best.pt and runs the criterion once
on real training-fold batches.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_mask2former import collate
from scripts.train_mask2former_improved import build_improved
from tiger_models.data import LabelMap, TigerMultitaskDataset, records_for_fold


def main():
    experiment_id = sys.argv[1]
    config = yaml.safe_load((ROOT / f"configs/{experiment_id}.yaml").read_text())
    labelmap = LabelMap.load(ROOT / "data/labelmap.csv")
    config["fine_to_coarse"] = labelmap.fine_to_coarse.astype(int).tolist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_improved(config, labelmap).to(device)
    ckpt_path = ROOT / f"artifacts/dsml_phase0/{experiment_id}/fold_0/best.pt"
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"Loaded {ckpt_path} (epoch {state['epoch']})")

    train_records = records_for_fold(ROOT / "data", 0, "train")
    ds = TigerMultitaskDataset(train_records, labelmap, config["height"], config["width"], False)
    loader = DataLoader(ds, batch_size=int(config["batch_size"]), shuffle=False, collate_fn=collate)

    totals = []
    with torch.no_grad():
        for batch in loader:
            output = model.segmenter(
                pixel_values=batch["image"].to(device),
                mask_labels=[x.to(device) for x in batch["mask_labels"]],
                class_labels=[x.to(device) for x in batch["class_labels"]],
            )
            loss_dict = model.segmenter.criterion(
                masks_queries_logits=output.masks_queries_logits,
                class_queries_logits=output.class_queries_logits,
                mask_labels=[x.to(device) for x in batch["mask_labels"]],
                class_labels=[x.to(device) for x in batch["class_labels"]],
            )
            raw = float(loss_dict["loss_sink_margin"].detach())
            weighted = raw * model.segmenter.weight_dict.get("loss_sink_margin", 1.0)
            ce = float(loss_dict["loss_cross_entropy"].detach())
            totals.append((raw, weighted, ce))
            print(f"batch: loss_sink_margin(raw)={raw:.6f}  loss_cross_entropy={ce:.6f}")

    raws = [t[0] for t in totals]
    print(f"\nmean loss_sink_margin (raw, post sink_margin_weight already applied inside _one_image) "
          f"= {sum(raws)/len(raws):.6f}  min={min(raws):.6f}  max={max(raws):.6f}")


if __name__ == "__main__":
    main()
