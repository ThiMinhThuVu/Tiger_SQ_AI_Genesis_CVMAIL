#!/usr/bin/env python3
"""Held-out 5-fold test inference for the improved TIGER Mask2Former."""
from __future__ import annotations

import argparse, json, statistics, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_mask2former import collate
from scripts.train_mask2former_improved import build_improved
from tiger_models.data import LabelMap, TigerMultitaskDataset, records_for_fold
from tiger_models.improved_mask2former import (aggregate_fine_to_coarse,
    soft_presence_modulation)
from tiger_models.metrics import segmentation_failure_metrics, segmentation_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true",
                        help="Allow slow CPU inference for engineering checks")
    parser.add_argument("--folds", type=int, nargs="+", choices=range(5),
                        help="Evaluate only these folds; default evaluates all five")
    return parser.parse_args()


@torch.inference_mode()
def infer_fold(config, model_dir, fold, data_root, device, labelmap,
               records_provider=records_for_fold):
    checkpoint_path = model_dir / f"fold_{fold}" / "best.pt"
    if not checkpoint_path.is_file():
        return None
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_improved(config, labelmap).to(device)
    model.load_state_dict(checkpoint["model"], strict=True); model.eval()
    records = records_provider(data_root, fold, "test")
    dataset = TigerMultitaskDataset(records, labelmap, config["height"], config["width"], False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=2 if device.type == "cuda" else 0, collate_fn=collate,
                        pin_memory=device.type == "cuda")
    fine_predictions=[]; coarse_predictions=[]; targets=[]; cases=[]; names=[]
    for batch in loader:
        result = model(batch["image"].to(device))
        size = batch["fine"].shape[-2:]
        fine = F.interpolate(result["fine_probability"], size, mode="bilinear", align_corners=False)
        if config.get("presence_head", True):
            fine = soft_presence_modulation(fine, result["presence_logits"],
                float(config["presence_epsilon"]), float(config["presence_gamma"]))
        derived = aggregate_fine_to_coarse(fine, model.fine_to_coarse)
        if config.get("direct_coarse_head", True):
            direct = F.interpolate(result["coarse_logits"], size, mode="bilinear",
                                   align_corners=False).softmax(1)
            beta = float(config["coarse_fusion_beta"]); coarse = beta*direct + (1-beta)*derived
        else:
            coarse = derived
        fine_predictions.append(fine.argmax(1).cpu().numpy()[0])
        coarse_predictions.append(coarse.argmax(1).cpu().numpy()[0])
        targets.append(batch["fine"].numpy()[0]); cases.append(batch["case_id"][0]); names.append(batch["name"][0])
    fine = segmentation_metrics(fine_predictions, targets, cases, labelmap.fine_weights, labelmap.fine_names)
    coarse_targets = [labelmap.fine_to_coarse[target] for target in targets]
    coarse = segmentation_metrics(coarse_predictions, coarse_targets, cases,
                                  labelmap.coarse_weights, labelmap.coarse_names)
    failures = segmentation_failure_metrics(fine_predictions, targets, labelmap.fine_names)
    task1_score = float(np.mean([fine["dice"], 1-fine["nhd"]]))
    surrogate = float(np.mean([fine["dice"], 1-fine["nhd"], coarse["dice"], 1-coarse["nhd"]]))
    return {"fold":fold, "checkpoint":str(checkpoint_path), "checkpoint_epoch":checkpoint["epoch"],
            "test_frames":len(records), "test_cases":sorted(set(cases)), "names":names,
            "task1_score":task1_score, "task12_surrogate":surrogate,
            "fine_dice":fine["dice"], "fine_nhd":fine["nhd"],
            "coarse_dice":coarse["dice"], "coarse_nhd":coarse["nhd"],
            "fine":fine, "coarse":coarse, "failure_metrics":failures}


def main():
    args=parse_args(); config=yaml.safe_load(args.config.read_text())
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("Held-out test inference requires CUDA; use --allow-cpu for a slow direct run")
    labelmap=LabelMap.load(args.data_root/"labelmap.csv")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested_folds = sorted(set(args.folds)) if args.folds else list(range(5))
    folds=[]
    for fold in requested_folds:
        result=infer_fold(config,args.model_dir,fold,args.data_root,device,labelmap)
        if result is not None: folds.append(result)
    if len(folds)!=len(requested_folds) and not args.allow_incomplete:
        raise RuntimeError(f"Only {len(folds)}/{len(requested_folds)} requested checkpoints available")
    metrics=("task1_score","task12_surrogate","fine_dice","fine_nhd","coarse_dice","coarse_nhd")
    summary={key:{"mean":statistics.mean(row[key] for row in folds),
                  "sample_standard_deviation":statistics.stdev(row[key] for row in folds) if len(folds)>1 else 0.}
             for key in metrics}
    selection_metric=str(config.get("selection_metric","task12_surrogate"))
    output={"experiment":config["experiment_id"],"split":"held_out_test",
            "requested_folds":requested_folds,"completed_test_folds":len(folds),
            "test_selection_policy":f"checkpoint selected on validation {selection_metric} only",
            "selection_score":None,"task12_surrogate_definition":"mean(fine Dice, 1-fine nHD, coarse Dice, 1-coarse nHD)",
            "task1_score_definition":"mean(fine Dice, 1-fine nHD)",
            "folds":folds,"summary":summary,
            "note":"No Task-3 head; official six-component selection_score is unavailable."}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2)+"\n"); print(json.dumps({"summary":summary},indent=2))

if __name__ == "__main__": main()
