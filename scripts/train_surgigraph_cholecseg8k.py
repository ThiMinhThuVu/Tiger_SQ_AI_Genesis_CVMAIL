#!/usr/bin/env python3
"""Train/evaluate SurgiGraph-Q A3-r2-GF on the fixed CholecSeg8k protocol."""
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
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_mask2former import collate
from tiger_models.improved_mask2former import ImprovedMask2Former
from tiger_models.metrics import segmentation_failure_metrics, segmentation_metrics


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, default=ROOT / "artifacts/cholecseg8k_surgigraph")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--split", choices=("train", "test"), default="train")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="Continue from best.pt and train until config epochs is reached")
    p.add_argument("--allow-cpu", action="store_true")
    return p.parse_args()


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def split_dir(root: Path, config: dict | None = None) -> Path:
    name = str((config or {}).get("split_dir", "grasp_splits_seed2026"))
    return root / name


def load_pairs(root: Path, fold: int, split: str, config: dict | None = None) -> list[dict]:
    if fold not in range(5):
        raise ValueError("fold must be 0..4")
    name = "test.json" if split == "test" else f"fold{fold}_{split}.json"
    pairs = json.loads((split_dir(root, config) / name).read_text())
    for pair in pairs:
        if not (root / pair["image"]).is_file() or not (root / pair["mask"]).is_file():
            raise FileNotFoundError(pair)
    return pairs


def audit_protocol(root: Path, config: dict) -> dict:
    meta = json.loads((split_dir(root, config) / "splits.json").read_text())
    test = load_pairs(root, 0, "test", config)
    test_images = {x["image"] for x in test}
    folds = []
    for fold in range(5):
        train = load_pairs(root, fold, "train", config); val = load_pairs(root, fold, "val", config)
        train_images = {x["image"] for x in train}; val_images = {x["image"] for x in val}
        if train_images & val_images or train_images & test_images or val_images & test_images:
            raise RuntimeError(f"image leakage in fold {fold}")
        train_videos = {Path(x).parts[0] for x in train_images}
        val_videos = {Path(x).parts[0] for x in val_images}
        test_videos = {Path(x).parts[0] for x in test_images}
        if train_videos & val_videos or train_videos & test_videos or val_videos & test_videos:
            raise RuntimeError(f"video leakage in fold {fold}")
        folds.append({"fold": fold, "train": len(train), "val": len(val)})
    if len(test) != int(meta["test_frames"]):
        raise RuntimeError("test frame count differs from frozen split metadata")
    return {"status": "PASS", "test_frames": len(test), "folds": folds}


class CholecDataset(Dataset):
    def __init__(self, root: Path, pairs: list[dict], height: int, width: int, training: bool):
        self.root, self.pairs = root, pairs
        self.height, self.width, self.training = height, width, training

    def __len__(self): return len(self.pairs)

    @staticmethod
    def uniform(low, high): return low + (high - low) * torch.rand(()).item()

    def __getitem__(self, index):
        pair = self.pairs[index]
        image = Image.open(self.root / pair["image"]).convert("RGB").resize(
            (self.width, self.height), Image.Resampling.BILINEAR)
        mask = Image.open(self.root / pair["mask"]).convert("L").resize(
            (self.width, self.height), Image.Resampling.NEAREST)
        if self.training:
            if torch.rand(()).item() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if torch.rand(()).item() < 0.75:
                image = ImageEnhance.Brightness(image).enhance(self.uniform(0.82, 1.18))
            if torch.rand(()).item() < 0.75:
                image = ImageEnhance.Contrast(image).enhance(self.uniform(0.85, 1.15))
            if torch.rand(()).item() < 0.5:
                image = ImageEnhance.Color(image).enhance(self.uniform(0.88, 1.12))
        fine = np.asarray(mask, dtype=np.int64)
        if fine.min(initial=0) < 0 or fine.max(initial=0) >= 13:
            raise ValueError(f"mask IDs outside 0..12: {pair['mask']}")
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = (array - np.asarray([.485, .456, .406], np.float32)) / np.asarray([.229, .224, .225], np.float32)
        video = Path(pair["image"]).parts[0]
        return {"image": torch.from_numpy(array.transpose(2, 0, 1).copy()),
                "fine": torch.from_numpy(fine.copy()), "visibility": torch.zeros(14),
                "name": pair["image"], "case_id": video}


def build_model(config: dict):
    from transformers import Mask2FormerForUniversalSegmentation
    names = list(config["class_names"]); n = int(config["num_labels"])
    base = Mask2FormerForUniversalSegmentation.from_pretrained(
        config["pretrained_checkpoint"], num_labels=n,
        id2label={i: x for i, x in enumerate(names)},
        label2id={x: i for i, x in enumerate(names)}, ignore_mismatched_sizes=True)
    options = {"epsilon": float(config["ot_epsilon"]), "iterations": int(config["ot_iterations"]),
        "target_mass_alpha": float(config["ot_target_mass_alpha"]),
        "target_mass_max": float(config["ot_target_mass_max"]),
        "dustbin_fp_weight": float(config["ot_dustbin_fp_weight"]),
        "pair_cost_risk_exponent": float(config["ot_pair_cost_risk_exponent"]),
        "dustbin_cost": float(config["ot_dustbin_cost"]), "mass_mode": str(config["ot_mass_mode"]),
        "convergence_tolerance": float(config["ot_convergence_tolerance"]),
        "check_interval": int(config["ot_sinkhorn_check_interval"]),
        "balance_iterations": int(config["ot_balance_iterations"]), "context_fp_strength": 0.0}
    return ImprovedMask2Former(base, torch.arange(n), torch.ones(n),
        matching_method="asymmetric_partial_sinkhorn", ot_options=options)


@torch.no_grad()
def evaluate(model, loader, device, config):
    model.eval(); predictions=[]; targets=[]; cases=[]
    for batch in loader:
        with torch.amp.autocast(device_type=device.type,
                                enabled=bool(config.get("amp", False) and device.type == "cuda")):
            result = model(batch["image"].to(device))
        probability = F.interpolate(result["fine_probability"], batch["fine"].shape[-2:],
                                    mode="bilinear", align_corners=False)
        predictions.extend(probability.argmax(1).cpu().numpy())
        targets.extend(batch["fine"].numpy()); cases.extend(batch["case_id"])
    names = list(config["class_names"]); weights = np.ones(len(names), dtype=np.float32)
    metric = segmentation_metrics(predictions, targets, cases, weights, names)
    confusion = np.zeros((len(names), len(names)), dtype=np.int64)
    for prediction, target in zip(predictions, targets):
        np.add.at(confusion, (target.ravel(), prediction.ravel()), 1)
    intersection = np.diag(confusion).astype(np.float64)
    union = confusion.sum(0) + confusion.sum(1) - intersection
    observable = confusion.sum(1) > 0
    per_class_iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    miou = float(per_class_iou[observable].mean())
    macro_dice = float(np.mean(2 * intersection[observable] /
                               np.maximum(confusion.sum(0)[observable] + confusion.sum(1)[observable], 1)))
    score = float(np.mean([metric["dice"], 1.0 - metric["nhd"]]))
    return {"task1_score": score, "mIoU": miou, "macro_dice": macro_dice,
            "per_class_IoU": per_class_iou.tolist(),
            "fine_dice": metric["dice"], "fine_nhd": metric["nhd"],
            "fine": metric, "failure_metrics": segmentation_failure_metrics(predictions, targets, names)}


def train(args, config, device):
    fold=args.fold; seed_all(int(config["seed"]) + fold)
    train_pairs=load_pairs(args.data_root, fold, "train", config)
    val_pairs=load_pairs(args.data_root, fold, "val", config)
    if args.smoke:
        train_pairs=train_pairs[:max(4, int(config["batch_size"]))]; val_pairs=val_pairs[:2]
    train_ds=CholecDataset(args.data_root, train_pairs, config["height"], config["width"], True)
    val_ds=CholecDataset(args.data_root, val_pairs, config["height"], config["width"], False)
    kwargs={"num_workers": int(config["num_workers"]), "collate_fn": collate, "pin_memory": device.type == "cuda"}
    train_loader=DataLoader(train_ds, batch_size=int(config["batch_size"]), shuffle=True, **kwargs)
    val_loader=DataLoader(val_ds, batch_size=1, shuffle=False, **kwargs)
    model=build_model(config).to(device)
    optimizer=torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    amp=bool(config.get("amp", False) and device.type == "cuda")
    scaler=torch.amp.GradScaler("cuda", enabled=amp)
    out=args.output_root/config["experiment_id"]/f"fold_{fold}"; out.mkdir(parents=True, exist_ok=True)
    if (out/"completion.json").exists():
        print(json.dumps({"status": "already_completed", "fold": fold, "output": str(out)}), flush=True)
        return
    epochs=1 if args.smoke else int(config["epochs"]); best=-float("inf"); bad=0; history=[]; start_epoch=0
    checkpoint_path = out / "best.pt"
    if args.resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"resume requested but checkpoint is missing: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        checkpoint_metrics = checkpoint.get("metrics", {})
        best = float(checkpoint_metrics.get("val_task1_score", -float("inf")))
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint and amp:
            scaler.load_state_dict(checkpoint["scaler"])
        print(json.dumps({"status": "resumed", "fold": fold, "start_epoch": start_epoch,
                          "target_epochs": epochs, "optimizer_restored": "optimizer" in checkpoint}), flush=True)
    for epoch in range(start_epoch, epochs):
        model.train(); optimizer.zero_grad(set_to_none=True); losses=[]
        for step,batch in enumerate(train_loader):
            micro = int(config.get("micro_batch_size", config["batch_size"]))
            chunks = (len(batch["image"]) + micro - 1) // micro
            step_losses=[]
            for start in range(0, len(batch["image"]), micro):
                stop=min(start+micro,len(batch["image"]))
                with torch.amp.autocast(device_type=device.type, enabled=amp):
                    result=model(batch["image"][start:stop].to(device),
                                 [x.to(device) for x in batch["mask_labels"][start:stop]],
                                 [x.to(device) for x in batch["class_labels"][start:stop]])
                    raw_loss=result["output"].loss
                scaler.scale(raw_loss/(chunks*int(config["gradient_accumulation"]))).backward()
                step_losses.append(float(raw_loss.detach()))
            losses.append(float(np.mean(step_losses)))
            if (step+1)%int(config["gradient_accumulation"])==0 or step+1==len(train_loader):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        metric=evaluate(model,val_loader,device,config)
        row={"epoch":epoch,"train_loss":float(np.mean(losses)),**{f"val_{k}":v for k,v in metric.items() if isinstance(v,float)}}
        history.append(row); print(json.dumps(row),flush=True)
        if metric["task1_score"] > best + float(config["early_stopping_min_delta"]):
            best=metric["task1_score"]; bad=0
            torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),
                        "scaler":scaler.state_dict(),"config":config,"epoch":epoch,"metrics":row},out/"best.pt")
            (out/"best_validation_metrics.json").write_text(json.dumps(metric,indent=2)+"\n")
        else: bad+=1
        if bad>=int(config["early_stopping_patience"]): break
    (out/"epoch_metrics.json").write_text(json.dumps(history,indent=2)+"\n")
    (out/"completion.json").write_text(json.dumps({"status":"smoke" if args.smoke else "completed","fold":fold,"epochs_run":len(history)},indent=2)+"\n")


def test(args, config, device):
    model=build_model(config).to(device); ckpt=args.output_root/config["experiment_id"]/f"fold_{args.fold}"/"best.pt"
    model.load_state_dict(torch.load(ckpt,map_location=device,weights_only=False)["model"])
    ds=CholecDataset(args.data_root,load_pairs(args.data_root,args.fold,"test",config),config["height"],config["width"],False)
    loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=int(config["num_workers"]),collate_fn=collate,pin_memory=device.type=="cuda")
    result=evaluate(model,loader,device,config)
    out=args.output_root/config["experiment_id"]/f"test_fold{args.fold}.json"
    if out.exists(): raise FileExistsError(f"refusing to overwrite {out}")
    out.write_text(json.dumps({"fold":args.fold,"split":"fixed_20_percent_video_test",**result},indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if isinstance(v,float)},indent=2))


def main():
    args=arguments(); config=yaml.safe_load(args.config.read_text()); print(json.dumps(audit_protocol(args.data_root, config)))
    if not torch.cuda.is_available() and not args.allow_cpu: raise RuntimeError("CUDA unavailable")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (train if args.split=="train" else test)(args,config,device)


if __name__=="__main__": main()
