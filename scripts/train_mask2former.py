#!/usr/bin/env python3
"""Mask2Former semantic-segmentation baseline runner.

This first baseline evaluates the 31-class fine segmentation task. Visibility
is not predicted by the stock Mask2Former head, so selection_score is left
explicitly unavailable until a multitask visibility head is added.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from tiger_models.data import LabelMap, TigerMultitaskDataset, records_for_fold
from tiger_models.metrics import normalized_hausdorff, binary_dice
from tiger_models.monai_mask2former_loss import (
    build_monai_dice_focal_loss, mask2former_monai_loss,
)


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--run-all-folds", action="store_true")
    p.add_argument("--smoke", action="store_true", help="run one epoch on fold 0")
    p.add_argument("--output-root", type=Path, default=ROOT / "artifacts/mask2former")
    p.add_argument("--data-root", type=Path, default=ROOT / "data")
    p.add_argument("--allow-cpu", action="store_true")
    return p.parse_args()


def seed(value: int) -> None:
    random.seed(value); np.random.seed(value); torch.manual_seed(value)


def build_model(config: dict):
    try:
        from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation, ResNetConfig
    except ImportError as exc:
        raise RuntimeError("Install transformers>=4.50 and accelerate before running Mask2Former") from exc
    if config.get("backbone") == "resnet50":
        backbone = ResNetConfig(
            depths=[3, 4, 6, 3], hidden_sizes=config.get("backbone_hidden_sizes", [256, 512, 1024, 2048]),
            out_features=["stage1", "stage2", "stage3", "stage4"],
        )
        model = Mask2FormerForUniversalSegmentation(Mask2FormerConfig(
            num_labels=int(config["num_labels"]), backbone_config=backbone,
        ))
        checkpoint = config.get("official_detectron2_checkpoint")
        if checkpoint:
            with Path(checkpoint).open("rb") as handle:
                source = pickle.load(handle, encoding="latin1")["model"]
            target = model.state_dict()
            converted = {}
            for key, value in source.items():
                if not key.startswith("backbone."):
                    continue
                name = key[len("backbone."):]
                if name.startswith("stem."):
                    suffix = name[len("stem."):]
                    suffix = suffix.replace("conv1.", "convolution.", 1).replace("norm.", "normalization.", 1)
                    name = "embedder.embedder." + suffix
                else:
                    parts = name.split(".")
                    stage = {"res2": 0, "res3": 1, "res4": 2, "res5": 3}.get(parts[0])
                    if stage is None:
                        continue
                    block = int(parts[1]); suffix = ".".join(parts[2:])
                    if suffix.startswith("shortcut."):
                        suffix = suffix.replace("shortcut.weight", "shortcut.convolution.weight")
                        suffix = suffix.replace("shortcut.norm.", "shortcut.normalization.")
                    else:
                        for source, target_name in (("conv1", "layer.0"), ("conv2", "layer.1"), ("conv3", "layer.2")):
                            if suffix.startswith(source + "."):
                                suffix = target_name + "." + suffix[len(source) + 1:]
                                break
                        suffix = suffix.replace(".norm.", ".normalization.")
                        suffix = suffix.replace(".weight", ".convolution.weight", 1) if suffix.endswith(".weight") else suffix
                    name = f"encoder.stages.{stage}.layers.{block}.{suffix}"
                full = "model.pixel_level_module.encoder." + name
                if name.startswith("encoder."):
                    full = "model.pixel_level_module.encoder." + name
                if full in target and tuple(target[full].shape) == tuple(value.shape):
                    converted[full] = torch.from_numpy(np.asarray(value))
            model.load_state_dict(converted, strict=False)
            print(f"Loaded {len(converted)} official R50 backbone tensors from {checkpoint}", flush=True)
        return model
    checkpoint = config.get("pretrained_checkpoint")
    if not checkpoint:
        raise RuntimeError("Missing pretrained checkpoint for non-ResNet Mask2Former")
    return Mask2FormerForUniversalSegmentation.from_pretrained(
        checkpoint, num_labels=int(config["num_labels"]), ignore_mismatched_sizes=True
    )


def collate(batch):
    images = torch.stack([item["image"] for item in batch])
    masks, labels = [], []
    for item in batch:
        target = item["fine"].long()
        classes = torch.unique(target)
        masks.append(torch.stack([(target == cls).float() for cls in classes]))
        labels.append(classes)
    return {"image": images, "fine": torch.stack([x["fine"] for x in batch]),
            "visibility": torch.stack([x["visibility"] for x in batch]),
            "mask_labels": masks, "class_labels": labels,
            "case_id": [x["case_id"] for x in batch], "name": [x["name"] for x in batch]}


def semantic_logits(output):
    class_prob = output.class_queries_logits.softmax(-1)[..., :-1]
    mask_prob = output.masks_queries_logits.sigmoid()
    return torch.einsum("bqc,bqhw->bchw", class_prob, mask_prob)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); dices=[]; nhds=[]; correct=0; pixels=0; losses=[]
    for batch in loader:
        image = batch["image"].to(device)
        out = model(pixel_values=image, mask_labels=[x.to(device) for x in batch["mask_labels"]],
                    class_labels=[x.to(device) for x in batch["class_labels"]])
        losses.append(float(out.loss.detach().cpu()))
        semantic = torch.nn.functional.interpolate(
            semantic_logits(out), size=batch["fine"].shape[-2:], mode="bilinear", align_corners=False
        )
        pred = semantic.argmax(1).cpu().numpy()
        target = batch["fine"].numpy()
        correct += int((pred == target).sum()); pixels += target.size
        for p, t in zip(pred, target):
            dices.append(binary_dice(p != 0, t != 0)); nhds.append(normalized_hausdorff(p != 0, t != 0))
    return {"loss": float(np.mean(losses)), "fine_dice": float(np.mean(dices)),
            "fine_nhd": float(np.mean(nhds)), "fine_pixel_accuracy": correct / pixels,
            "selection_score": None}


def run_fold(config, fold, args_ns):
    if not torch.cuda.is_available() and not args_ns.allow_cpu:
        raise RuntimeError("CUDA unavailable; use --allow-cpu only for smoke engineering checks")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed(int(config["seed"]) + fold)
    labelmap = LabelMap.load(args_ns.data_root / "labelmap.csv")
    train = TigerMultitaskDataset(records_for_fold(args_ns.data_root, fold, "train"), labelmap,
                                  config["height"], config["width"], training=True)
    val = TigerMultitaskDataset(records_for_fold(args_ns.data_root, fold, "validation"), labelmap,
                                config["height"], config["width"], training=False)
    kwargs = {"batch_size": config["batch_size"], "num_workers": config["num_workers"],
              "collate_fn": collate, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train, shuffle=True, **kwargs)
    val_loader = DataLoader(val, shuffle=False, **{**kwargs, "batch_size": 1})
    model = build_model(config).to(device)
    monai_weight = float(config.get("monai_aux_loss_weight", 0.0))
    monai_criterion = build_monai_dice_focal_loss() if monai_weight > 0 else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]),
                                  weight_decay=float(config["weight_decay"]))
    epochs = 1 if args_ns.smoke else int(config["epochs"])
    out_dir = args_ns.output_root / config["experiment_id"] / f"fold_{fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    history=[]; best=float("inf")
    for epoch in range(epochs):
        model.train(); running=[]; running_native=[]; running_monai=[]; optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            output = model(pixel_values=batch["image"].to(device),
                           mask_labels=[x.to(device) for x in batch["mask_labels"]],
                           class_labels=[x.to(device) for x in batch["class_labels"]])
            native_loss = output.loss
            auxiliary_loss = (
                mask2former_monai_loss(output, batch["fine"].to(device), monai_criterion)
                if monai_criterion is not None else native_loss.new_zeros(())
            )
            total_loss = native_loss + monai_weight * auxiliary_loss
            (total_loss / int(config["gradient_accumulation"])).backward()
            running.append(float(total_loss.detach().cpu()))
            running_native.append(float(native_loss.detach().cpu()))
            running_monai.append(float(auxiliary_loss.detach().cpu()))
            if (step + 1) % int(config["gradient_accumulation"]) == 0 or step + 1 == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); optimizer.zero_grad(set_to_none=True)
        metrics = evaluate(model, val_loader, device)
        record = {"epoch": epoch, "train_loss": float(np.mean(running)),
                  "train_native_loss": float(np.mean(running_native)),
                  "train_monai_aux_loss": float(np.mean(running_monai)),
                  **{f"val_{k}": v for k,v in metrics.items()}}
        history.append(record); print(json.dumps(record), flush=True)
        if metrics["loss"] < best:
            best = metrics["loss"]
            torch.save({"model": model.state_dict(), "config": config, "epoch": epoch, "metrics": record}, out_dir / "best.pt")
    (out_dir / "epoch_metrics.json").write_text(json.dumps(history, indent=2))
    (out_dir / "completion.json").write_text(json.dumps({"status": "smoke" if args_ns.smoke else "completed", "fold": fold, "epochs": epochs}, indent=2))


def main() -> int:
    ns = args(); config = yaml.safe_load(ns.config.read_text())
    if ns.run_all_folds:
        folds = range(5)
    else:
        folds = [0 if ns.smoke else ns.fold]
    for fold in folds:
        run_fold(config, fold, ns)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
