#!/usr/bin/env python3
"""Train/evaluate a 16-class model using only native coarse-mask targets."""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from full_version.task2_fine_31cls.scripts.data_full import (
    build_manifest, build_trainval_manifest, load_manifest, records_for_manifest,
)
from scripts.train_mask2former import build_model, semantic_logits
from tiger_models.data import LabelMap
from tiger_models.metrics import segmentation_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("train", "evaluate", "aggregate"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


class CoarseDataset(Dataset):
    def __init__(self, records, labelmap, height: int, width: int, training: bool):
        self.records = list(records); self.labelmap = labelmap
        self.height = int(height); self.width = int(width); self.training = training

    def __len__(self): return len(self.records)

    @staticmethod
    def _uniform(low, high): return low + (high - low) * torch.rand(()).item()

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(record.image) as handle: image = handle.convert("RGB")
        with Image.open(self.records[index].fine_mask.parent.parent / "masks_coarse" / record.name) as handle:
            ids = self.labelmap.decode_coarse(np.asarray(handle.convert("RGB")))
        mask = Image.fromarray(ids, mode="L")
        image = image.resize((self.width, self.height), Image.Resampling.BILINEAR)
        mask = mask.resize((self.width, self.height), Image.Resampling.NEAREST)
        if self.training:
            if torch.rand(()).item() < 0.75:
                from torchvision.transforms import InterpolationMode
                from torchvision.transforms.functional import affine
                angle = self._uniform(-10, 10)
                translation = [round(self._uniform(-.05, .05) * self.width), round(self._uniform(-.05, .05) * self.height)]
                scale = self._uniform(.90, 1.10)
                image = affine(image, angle, translation, scale, [0., 0.], interpolation=InterpolationMode.BILINEAR, fill=0)
                mask = affine(mask, angle, translation, scale, [0., 0.], interpolation=InterpolationMode.NEAREST, fill=0)
            if torch.rand(()).item() < .75: image = ImageEnhance.Brightness(image).enhance(self._uniform(.82, 1.18))
            if torch.rand(()).item() < .75: image = ImageEnhance.Contrast(image).enhance(self._uniform(.85, 1.15))
            if torch.rand(()).item() < .50: image = ImageEnhance.Color(image).enhance(self._uniform(.88, 1.12))
        array = np.asarray(image, dtype=np.float32) / 255.
        array = (array - np.asarray([.485, .456, .406], np.float32)) / np.asarray([.229, .224, .225], np.float32)
        target = np.asarray(mask, dtype=np.int64)
        return {"image": torch.from_numpy(array.transpose(2, 0, 1).copy()),
                "target": torch.from_numpy(target.copy()), "case_id": record.case_id, "name": record.name}


def collate(batch):
    targets = [row["target"].long() for row in batch]
    classes = [torch.unique(target) for target in targets]
    return {"image": torch.stack([row["image"] for row in batch]), "target": torch.stack(targets),
            "mask_labels": [torch.stack([(target == cls).float() for cls in labels]) for target, labels in zip(targets, classes)],
            "class_labels": classes, "case_id": [row["case_id"] for row in batch], "name": [row["name"] for row in batch]}


def context(ns):
    config = yaml.safe_load(ns.config.read_text()); manifest = load_manifest(ns.split_manifest)
    builders = {"full40_case_folds_v1": build_manifest,
                "full40_center_stratified_5fold_trainval_v1": build_trainval_manifest,
                "full40_center_stratified_5fold_trainval_d517_v2": build_trainval_manifest}
    if manifest.get("version") not in builders:
        raise ValueError(f"Unsupported manifest version: {manifest.get('version')}")
    current = builders[manifest["version"]](ns.data_root, int(config["split_seed"]))
    if current["snapshot"] != manifest.get("snapshot"):
        raise ValueError("Dataset does not match the frozen full40 manifest")
    return config, manifest, LabelMap.load(ns.data_root / "labelmap.csv")


def loader(ns, config, manifest, labelmap, fold, split, training=False):
    records = records_for_manifest(ns.data_root, manifest, fold, split)
    dataset = CoarseDataset(records, labelmap, config["height"], config["width"], training)
    return DataLoader(dataset, batch_size=config["batch_size"] if training else 1, shuffle=training,
                      num_workers=config["num_workers"], pin_memory=True, collate_fn=collate)


@torch.no_grad()
def evaluate_model(model, data_loader, device, labelmap, include_rows=False):
    model.eval(); predictions=[]; targets=[]; cases=[]; names=[]; losses=[]
    for batch in data_loader:
        output = model(pixel_values=batch["image"].to(device),
                       mask_labels=[x.to(device) for x in batch["mask_labels"]],
                       class_labels=[x.to(device) for x in batch["class_labels"]])
        losses.append(float(output.loss.detach().cpu()))
        logits = torch.nn.functional.interpolate(semantic_logits(output), size=batch["target"].shape[-2:], mode="bilinear", align_corners=False)
        predictions.extend(logits.argmax(1).cpu().numpy()); targets.extend(batch["target"].numpy())
        cases.extend(batch["case_id"]); names.extend(batch["name"])
    metrics = segmentation_metrics(predictions, targets, cases, labelmap.coarse_weights, labelmap.coarse_names)
    result = {"loss": float(np.mean(losses)), "dice": metrics["dice"], "nhd": metrics["nhd"],
              "task_score": (metrics["dice"] + 1. - metrics["nhd"]) / 2.,
              "per_case_dice": metrics["per_case_dice"], "per_case_nhd": metrics["per_case_nhd"]}
    center_scores = []
    for center_id in sorted({case_id.split("_case_", 1)[0] for case_id in set(cases)}):
        selected = {case_id for case_id in set(cases) if case_id.startswith(center_id + "_case_")}
        dice = statistics.mean(metrics["per_case_dice"][case_id] for case_id in selected)
        nhd = statistics.mean(metrics["per_case_nhd"][case_id] for case_id in selected)
        center_scores.append((dice + 1. - nhd) / 2.)
    result["center_macro_task_score"] = statistics.mean(center_scores)
    if include_rows: result.update({"names": names, "per_class": metrics["per_class"]})
    return result


def train(ns):
    if ns.fold is None: raise ValueError("--fold is required")
    config, manifest, labelmap = context(ns); device = torch.device("cuda")
    value = int(config["seed"]) + ns.fold; random.seed(value); np.random.seed(value); torch.manual_seed(value)
    out = ns.output_root / config["experiment_id"] / f"fold_{ns.fold}"
    if (out / "completion.json").exists(): raise FileExistsError(f"Completed run exists: {out}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_provenance.json").write_text(json.dumps({"task":"official_task1_coarse_only", "fold":ns.fold,
        "effective_seed":value, "manifest_version":manifest["version"], "uses_task2_checkpoint":False,
        "uses_fine_predictions":False, "target":"masks_coarse"}, indent=2)+"\n")
    train_loader = loader(ns, config, manifest, labelmap, ns.fold, "train", True)
    val_loader = loader(ns, config, manifest, labelmap, ns.fold, "validation")
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    epochs = int(config["smoke_epochs"] if ns.smoke else config["epochs"])
    best=-float("inf"); bad=0; history=[]
    for epoch in range(epochs):
        model.train(); running=[]; optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            output = model(pixel_values=batch["image"].to(device), mask_labels=[x.to(device) for x in batch["mask_labels"]], class_labels=[x.to(device) for x in batch["class_labels"]])
            (output.loss / int(config["gradient_accumulation"])).backward(); running.append(float(output.loss.detach().cpu()))
            if (step+1) % int(config["gradient_accumulation"]) == 0 or step+1 == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step(); optimizer.zero_grad(set_to_none=True)
        val = evaluate_model(model, val_loader, device, labelmap)
        row={"epoch":epoch,"train_loss":float(np.mean(running)),**{f"val_{k}":v for k,v in val.items() if not k.startswith("per_case")}}
        history.append(row); print(json.dumps(row), flush=True)
        selection_metric = str(config.get("selection_metric", "task_score"))
        if selection_metric not in val:
            raise KeyError(f"Unknown selection_metric {selection_metric!r}")
        if val[selection_metric] > best + float(config["early_stopping_min_delta"]):
            best=val[selection_metric]; bad=0
            torch.save({"model":model.state_dict(),"config":config,"epoch":epoch,"metrics":row}, out/"best.pt")
        else: bad += 1
        if not ns.smoke and bad >= int(config["early_stopping_patience"]): break
    (out/"epoch_metrics.json").write_text(json.dumps(history,indent=2)+"\n")
    (out/"completion.json").write_text(json.dumps({"status":"smoke" if ns.smoke else "completed","fold":ns.fold,"epochs_run":len(history)},indent=2)+"\n")


def evaluate_fold(ns):
    if ns.fold is None: raise ValueError("--fold is required")
    config, manifest, labelmap=context(ns); device=torch.device("cuda")
    checkpoint=ns.output_root/config["experiment_id"]/f"fold_{ns.fold}"/"best.pt"
    model=build_model(config); model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=False)["model"]); model.to(device)
    result=evaluate_model(model,loader(ns,config,manifest,labelmap,ns.fold,"test"),device,labelmap,True)
    result.update({"fold":ns.fold,"test_cases":manifest["folds"][str(ns.fold)]["test_cases"],"manifest_version":manifest["version"]})
    path=ns.output_root/config["experiment_id"]/"test_oof"/"folds"/f"fold_{ns.fold}.json"; path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+"\n")


def aggregate(ns):
    config, manifest, _=context(ns); base=ns.output_root/config["experiment_id"]/"test_oof"
    folds=[json.loads((base/"folds"/f"fold_{i}.json").read_text()) for i in range(5)]
    dice={}; nhd={}; names=[]
    for fold in folds:
        names += fold["names"]
        for case,value in fold["per_case_dice"].items():
            if case in dice: raise ValueError("Duplicate OOF case")
            dice[case]=value; nhd[case]=fold["per_case_nhd"][case]
    if len(dice)!=40 or len(names)!=524 or len(set(names))!=524: raise ValueError("Incomplete OOF coverage")
    d=statistics.mean(dice.values()); h=statistics.mean(nhd.values())
    payload={"status":"completed","official_task":"task1_merged_coarse_only","cases":40,"frames":524,
             "uses_task2_checkpoint":False,"uses_fine_predictions":False,"folds":folds,
             "overall":{"dice":d,"nhd":h,"task_score":(d+1.-h)/2.}}
    (base/"oof_test_metrics.json").write_text(json.dumps(payload,indent=2)+"\n")
    (base/"completion.json").write_text(json.dumps({"status":"completed","folds":5,"cases":40,"frames":524},indent=2)+"\n")
    print(json.dumps(payload["overall"],indent=2))


if __name__ == "__main__":
    ns=parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    {"train":train,"evaluate":evaluate_fold,"aggregate":aggregate}[ns.mode](ns)
