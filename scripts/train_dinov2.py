#!/usr/bin/env python3
"""DINOv2 ViT-S/14 semantic segmentation baseline for TIGER."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tiger_models.data import LabelMap, TigerMultitaskDataset, records_for_fold
from tiger_models.metrics import binary_dice, normalized_hausdorff


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--run-all-folds", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--data-root", type=Path, default=ROOT / "data")
    p.add_argument("--output-root", type=Path, default=ROOT / "artifacts/new_baselines")
    p.add_argument("--allow-cpu", action="store_true")
    return p.parse_args()


def set_seed(value: int) -> None:
    random.seed(value); np.random.seed(value); torch.manual_seed(value)


class DINOv2Segmenter(nn.Module):
    def __init__(self, checkpoint: str, num_labels: int, height: int, width: int, freeze: bool):
        super().__init__()
        from transformers import Dinov2Model
        self.encoder = Dinov2Model.from_pretrained(checkpoint)
        hidden = int(self.encoder.config.hidden_size)
        self.height, self.width = height, width
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden, 256, 3, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, num_labels, 1),
        )
        if freeze:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        output = self.encoder(pixel_values=image)
        tokens = output.last_hidden_state[:, 1:]
        patch = int(self.encoder.config.patch_size)
        gh, gw = self.height // patch, self.width // patch
        if tokens.shape[1] != gh * gw:
            raise RuntimeError(f"Unexpected DINOv2 patch grid: {tokens.shape[1]} != {gh}*{gw}")
        feature = tokens.transpose(1, 2).reshape(image.shape[0], -1, gh, gw)
        return F.interpolate(self.decoder(feature), size=image.shape[-2:], mode="bilinear", align_corners=False)


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = logits.softmax(1)
    one_hot = F.one_hot(target, logits.shape[1]).permute(0, 3, 1, 2).float()
    intersection = (probability * one_hot).sum((0, 2, 3))
    denominator = probability.sum((0, 2, 3)) + one_hot.sum((0, 2, 3))
    return 1.0 - ((2 * intersection + 1e-5) / (denominator + 1e-5)).mean()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); losses=[]; dices=[]; nhds=[]; correct=0; pixels=0
    for batch in loader:
        image, target = batch["image"].to(device), batch["fine"].to(device)
        logits = model(image)
        loss = F.cross_entropy(logits, target) + dice_loss(logits, target)
        pred = logits.argmax(1).cpu().numpy(); truth = target.cpu().numpy()
        losses.append(float(loss.cpu())); correct += int((pred == truth).sum()); pixels += truth.size
        for p, t in zip(pred, truth):
            dices.append(binary_dice(p != 0, t != 0)); nhds.append(normalized_hausdorff(p != 0, t != 0))
    return {"loss": float(np.mean(losses)), "fine_dice": float(np.mean(dices)),
            "fine_nhd": float(np.mean(nhds)), "fine_pixel_accuracy": correct / pixels}


def run_fold(config, fold, ns):
    if not torch.cuda.is_available() and not ns.allow_cpu:
        raise RuntimeError("CUDA unavailable; use --allow-cpu only for engineering smoke")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int(config["seed"]) + fold)
    labelmap = LabelMap.load(ns.data_root / "labelmap.csv")
    train = TigerMultitaskDataset(records_for_fold(ns.data_root, fold, "train"), labelmap,
                                  config["height"], config["width"], training=True)
    val = TigerMultitaskDataset(records_for_fold(ns.data_root, fold, "validation"), labelmap,
                                config["height"], config["width"], training=False)
    kwargs = {"num_workers": config["num_workers"], "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train, batch_size=config["batch_size"], shuffle=True, **kwargs)
    val_loader = DataLoader(val, batch_size=1, shuffle=False, **kwargs)
    model = DINOv2Segmenter(config["pretrained_checkpoint"], config["num_labels"],
                            config["height"], config["width"], config["freeze_encoder"]).to(device)
    encoder_parameters = [p for p in model.encoder.parameters() if p.requires_grad]
    decoder_parameters = list(model.decoder.parameters())
    groups = [{"params": decoder_parameters, "lr": float(config["learning_rate"])}]
    if encoder_parameters:
        groups.append({"params": encoder_parameters, "lr": float(config["encoder_learning_rate"])})
    optimizer = torch.optim.AdamW(groups, weight_decay=float(config["weight_decay"]))
    epochs = 1 if ns.smoke else int(config["epochs"])
    out_dir = ns.output_root / config["experiment_id"] / f"fold_{fold}"; out_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf"); history=[]
    for epoch in range(epochs):
        model.train(); running=[]; optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            target = batch["fine"].to(device)
            logits = model(batch["image"].to(device))
            loss = F.cross_entropy(logits, target) + dice_loss(logits, target)
            (loss / int(config["gradient_accumulation"])).backward()
            running.append(float(loss.detach().cpu()))
            if (step + 1) % int(config["gradient_accumulation"]) == 0 or step + 1 == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); optimizer.zero_grad(set_to_none=True)
        metrics = evaluate(model, val_loader, device)
        record = {"epoch": epoch, "train_loss": float(np.mean(running)), **{f"val_{k}": v for k,v in metrics.items()}}
        history.append(record); print(json.dumps(record), flush=True)
        if metrics["loss"] < best:
            best = metrics["loss"]
            torch.save({"model": model.state_dict(), "config": config, "epoch": epoch, "metrics": record}, out_dir / "best.pt")
    (out_dir / "epoch_metrics.json").write_text(json.dumps(history, indent=2))
    (out_dir / "completion.json").write_text(json.dumps({"status": "smoke" if ns.smoke else "completed", "fold": fold, "epochs": epochs}, indent=2))


def main():
    ns = parse_args(); config = yaml.safe_load(ns.config.read_text())
    folds = range(5) if ns.run_all_folds else [0 if ns.smoke else ns.fold]
    for fold in folds: run_fold(config, fold, ns)


if __name__ == "__main__":
    main()
