#!/usr/bin/env python3
"""GeoSurg-IC decoder adaptation for the Task-1 16-class coarse-only model.

Same recipe as run_pilot.py (Task 2), transferred to the D517 coarse-only
Mask2Former: frozen Swin encoder, fixed-step decoder fine-tuning, four-view
deterministic replay of the mixed response on 16-class semantic margins.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from full_version.geosurg_ic.run_pilot import grad_norm, save_json, seed_all
from full_version.task1_merged_16cls.scripts.train_coarse_only import (
    CoarseDataset, collate, evaluate_model)
from full_version.task2_fine_31cls.scripts.data_full import (
    build_trainval_manifest, load_manifest, records_for_manifest)
from scripts.train_mask2former import build_model, semantic_logits
from tiger_models.data import LabelMap
from tiger_models.geosurg_ic import appearance_views, interface_probe, log_margin, mixed_response

BASE = ROOT / "full_version/center_stratified_32_8"
CONFIG = BASE / "configs/d517_task1_coarse_best.yaml"
MANIFEST = BASE / "splits/case_folds_d517.json"
DEPTH = ROOT / "full_version/geosurg_ic/depth_cache"
GRID = (160, 280)


def checkpoint_path(config, fold):
    return BASE / "artifacts" / config["experiment_id"] / f"fold_{fold}/best.pt"


def load_model(config, checkpoint):
    model = build_model(config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    return model


def center_macro(metrics):
    cases = metrics["per_case_dice"]
    centers = sorted({c.split("_case_")[0] for c in cases})
    rows = {}
    for center in centers:
        sel = [c for c in cases if c.startswith(center + "_case_")]
        d = statistics.mean(metrics["per_case_dice"][c] for c in sel)
        n = statistics.mean(metrics["per_case_nhd"][c] for c in sel)
        rows[center] = {"dice": d, "nhd": n, "score": (d + 1 - n) / 2}
    return rows


def validate(model, config, manifest, labelmap, fold):
    records = records_for_manifest(ROOT / "data", manifest, fold, "validation")
    ds = CoarseDataset(records, labelmap, config["height"], config["width"], False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate)
    seed_all(0)  # HF loss point sampling only affects the reported loss
    metrics = evaluate_model(model, loader, torch.device("cuda"), labelmap)
    metrics["per_center"] = center_macro(metrics)
    return metrics


def train(args, config, manifest, labelmap, out):
    seed_all(20260914 + args.fold)
    records = records_for_manifest(ROOT / "data", manifest, args.fold, "train")
    validation = records_for_manifest(ROOT / "data", manifest, args.fold, "validation")
    assert not ({r.case_id for r in records} & {r.case_id for r in validation})
    # No spatial augmentation: preserves alignment with the cached pseudo-depth.
    ds = CoarseDataset(records, labelmap, config["height"], config["width"], False)
    checkpoint = checkpoint_path(config, args.fold)
    model = load_model(config, checkpoint)
    frozen = []
    for name, p in model.named_parameters():
        if name.startswith("model.pixel_level_module.encoder."):
            p.requires_grad_(False)
            frozen.append(name)
    assert frozen, "Backbone freeze path did not match"
    model.cuda().eval()  # deterministic four-view replay, gradients still enabled
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(20260914 + args.fold)
    order = np.concatenate([rng.permutation(len(ds)) for _ in range(args.steps // len(ds) + 1)])
    provenance = {"args": vars(args) | {"output": str(args.output)}, "task": "task1_coarse_only_16cls",
        "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "manifest": str(MANIFEST),
        "train_cases": sorted({r.case_id for r in records}),
        "validation_cases": sorted({r.case_id for r in validation}),
        "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "device": torch.cuda.get_device_name(), "gpu_count": torch.cuda.device_count(),
        "config": config, "no_inference_change": True}
    save_json(out / "provenance.json", provenance)
    print(json.dumps({"event": "initialized", "arm": args.arm, "fold": args.fold,
                      "trainable_parameters": provenance["trainable_parameters"]}), flush=True)
    start = time.monotonic()
    history = []
    for step in range(args.steps):
        seed = 20260914 + args.fold * 100000 + step
        seed_all(seed)
        item = ds[int(order[step])]
        batch = collate([item])
        image = batch["image"].cuda()
        target = F.interpolate(batch["target"][:, None].float().cuda(), GRID, mode="nearest")[0, 0].long()
        with np.load(DEPTH / (item["name"] + ".npz")) as cached:
            depth = torch.from_numpy(cached["depth"].astype(np.float32)).cuda()
        probe = interface_probe(target, depth, seed)
        views = appearance_views(image, probe, seed + 123) if probe is not None else [image] * 4
        optimizer.zero_grad(set_to_none=True)
        # Identical images, probes and loss RNG in every arm; view cycles 0..3.
        seed_all(seed)
        output = model(pixel_values=views[step % 4],
                       mask_labels=[x.cuda() for x in batch["mask_labels"]],
                       class_labels=[x.cuda() for x in batch["class_labels"]])
        supervised = output.loss
        if not torch.isfinite(supervised):
            raise RuntimeError("Nonfinite supervised objective")
        supervised.backward()
        sup_value = float(supervised.detach())
        del output, supervised
        ic_value, ic_norm = 0.0, 0.0
        sup_norm = grad_norm(model) if args.audit else None
        do_ic = probe is not None and probe["pixels"] > 0 and step % args.ic_every == 0
        if do_ic:
            weights = probe["weights"]["geo"]
            with torch.no_grad():
                margins = [log_margin(semantic_logits(model(pixel_values=v)), probe, GRID) for v in views]
                delta = mixed_response(margins)
                ic_value = float((weights * delta.square()).sum())
                coefficient = 2 * weights * delta * args.ic_weight
            if args.arm == "geo":
                if args.audit:
                    supervised_grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()
                                        if p.grad is not None}
                    optimizer.zero_grad(set_to_none=True)
                for sign, view in zip((1, -1, -1, 1), views):
                    margin = log_margin(semantic_logits(model(pixel_values=view)), probe, GRID)
                    (sign * coefficient * margin).sum().backward()
                if args.audit:
                    ic_norm = grad_norm(model)
                    for n, p in model.named_parameters():
                        if n in supervised_grads:
                            if p.grad is None:
                                p.grad = supervised_grads[n]
                            else:
                                p.grad.add_(supervised_grads[n])
                    del supervised_grads
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        if not np.isfinite(norm):
            raise RuntimeError("Nonfinite gradient norm")
        optimizer.step()
        row = {"step": step + 1, "name": item["name"], "supervised": sup_value,
               "ic_raw": ic_value, "gradient_norm": norm,
               "routed_pixels": 0 if probe is None else probe["pixels"],
               "boundary_pixels": 0 if probe is None else probe["boundary_pixels"],
               "seconds": time.monotonic() - start,
               "peak_memory_mb": torch.cuda.max_memory_allocated() / 1024**2}
        if args.audit:
            row.update(supervised_grad_norm=sup_norm, weighted_ic_grad_norm=ic_norm if do_ic else 0)
        history.append(row)
        if step % 8 == 0 or step + 1 == args.steps:
            save_json(out / "history.json", history)
            print(json.dumps(row), flush=True)
        if time.monotonic() - start > args.max_seconds:
            save_json(out / "timeout.json", {"completed_steps": step + 1})
            raise TimeoutError("Per-run time budget exhausted")
    if args.audit:
        ratios = [r["weighted_ic_grad_norm"] / max(r["supervised_grad_norm"], 1e-12)
                  for r in history if r["weighted_ic_grad_norm"]]
        save_json(out / "completion.json", {"status": "audit_completed", "steps": args.steps,
            "ic_weight": args.ic_weight, "ic_to_supervised_grad_ratio": ratios,
            "median_ratio": float(np.median(ratios)) if ratios else None,
            "seconds": time.monotonic() - start})
        print(json.dumps({"median_ratio": float(np.median(ratios)) if ratios else None,
                          "ratios": ratios}), flush=True)
        return
    torch.save({"model": model.cpu().state_dict(), "config": config, "epoch": args.steps,
                "geosurg_ic": provenance}, out / "best.pt")
    model.cuda().eval()
    metrics = validate(model, config, manifest, labelmap, args.fold)
    save_json(out / "validation.json", {"grid": [config["height"], config["width"]],
        "split": "validation_used_for_original_checkpoint_selection", "metrics": metrics})
    save_json(out / "completion.json", {"status": "completed", "steps": args.steps,
                                        "seconds": time.monotonic() - start})
    print(json.dumps({"event": "completed", "arm": args.arm, "fold": args.fold,
        "metrics": {k: v for k, v in metrics.items() if isinstance(v, float)}}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["aug", "geo", "baseline"], required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--steps", type=int, default=192)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--ic-weight", type=float, default=0.02)
    p.add_argument("--ic-every", type=int, default=2)
    p.add_argument("--max-seconds", type=int, default=3600)
    p.add_argument("--audit", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    assert torch.cuda.device_count() == 1, "Exactly one GPU must be visible"
    torch.set_num_threads(4)
    config = yaml.safe_load(CONFIG.read_text())
    manifest = load_manifest(MANIFEST)
    current = build_trainval_manifest(ROOT / "data", int(config["split_seed"]))
    assert current["snapshot"] == manifest["snapshot"], "Dataset changed"
    labelmap = LabelMap.load(ROOT / "data/labelmap.csv")
    out = args.output / args.arm / f"fold_{args.fold}"
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    out.mkdir(parents=True)
    if args.arm == "baseline":
        checkpoint = checkpoint_path(config, args.fold)
        model = load_model(config, checkpoint).cuda().eval()
        metrics = validate(model, config, manifest, labelmap, args.fold)
        save_json(out / "validation.json", {"grid": [config["height"], config["width"]],
            "checkpoint": str(checkpoint), "metrics": metrics})
        save_json(out / "completion.json", {"status": "completed"})
        print(json.dumps({"arm": "baseline", "fold": args.fold, "center_macro": metrics["center_macro_task_score"]}))
        return
    train(args, config, manifest, labelmap, out)


if __name__ == "__main__":
    main()
