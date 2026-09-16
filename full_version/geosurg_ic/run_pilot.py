#!/usr/bin/env python3
"""One-GPU, checkpoint-initialized GeoSurg-IC deadline pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
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
from full_version.task2_fine_31cls.scripts.data_full import (
    build_trainval_manifest, load_manifest, records_for_manifest)
from scripts.train_mask2former import collate
from scripts.train_mask2former_improved import build_improved, evaluate, objective
from tiger_models.data import LabelMap, TigerMultitaskDataset
from tiger_models.geosurg_ic import appearance_views, interface_probe, log_margin, mixed_response
from tiger_models.improved_mask2former import presence_pos_weight, presence_targets

BASE = ROOT / "full_version/center_stratified_32_8"


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def grad_norm(model):
    return float(torch.stack([p.grad.detach().float().square().sum()
        for p in model.parameters() if p.grad is not None]).sum().sqrt())


def train(args, config, manifest, labelmap, out):
    seed_all(20260914 + args.fold)
    records = records_for_manifest(ROOT / "data", manifest, args.fold, "train")
    validation = records_for_manifest(ROOT / "data", manifest, args.fold, "validation")
    assert not ({r.case_id for r in records} & {r.case_id for r in validation})
    ds = TigerMultitaskDataset(records, labelmap, config["height"], config["width"], False)
    val_ds = TigerMultitaskDataset(validation, labelmap, config["height"], config["width"], False)
    checkpoint = BASE / "artifacts" / config["experiment_id"] / f"fold_{args.fold}/best.pt"
    model = build_improved(config, labelmap)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    del state
    frozen = []
    for name, p in model.named_parameters():
        if name.startswith("segmenter.model.pixel_level_module.encoder."):
            p.requires_grad_(False)
            frozen.append(name)
    assert frozen, "Backbone freeze path did not match"
    model.cuda().eval()  # deterministic four-view replay, gradients still enabled
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=args.lr, weight_decay=1e-4)
    # Tiny-grid presence is sufficient for weighting, but decode original labels
    # to retain tiny classes: use per-record original RGB masks here.
    from PIL import Image
    presence = []
    for r in records:
        with Image.open(r.fine_mask) as im:
            ids = labelmap.decode_fine(np.asarray(im.convert("RGB")))
        presence.append(torch.from_numpy(np.bincount(ids.ravel(), minlength=31) > 0).float())
    pos_weight = presence_pos_weight(torch.stack(presence), 8).cuda()
    station_weight = torch.ones(14, device="cuda")
    class_weight = torch.from_numpy(labelmap.fine_weights).cuda()
    rng = np.random.default_rng(20260914 + args.fold)
    order = np.concatenate([rng.permutation(len(ds)) for _ in range(args.steps // len(ds) + 1)])
    provenance = {"args": vars(args) | {"output": str(args.output)},
        "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "manifest": str(BASE / "splits/case_folds_d517.json"),
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
    size = (160, 280)
    for step in range(args.steps):
        seed = 20260914 + args.fold * 100000 + step
        seed_all(seed)
        item = ds[int(order[step])]
        batch = collate([item])
        batch["coarse"] = item["coarse"][None]
        image = batch["image"].cuda()
        target = F.interpolate(batch["fine"][:, None].float().cuda(), size, mode="nearest")[0, 0].long()
        cache = ROOT / "full_version/geosurg_ic/depth_cache" / (item["name"] + ".npz")
        with np.load(cache) as cached:
            depth = torch.from_numpy(cached["depth"].astype(np.float32)).cuda()
        probe = interface_probe(target, depth, seed)
        views = appearance_views(image, probe, seed + 123) if probe is not None else [image] * 4
        optimizer.zero_grad(set_to_none=True)
        # Every arm receives identical images/appearance probes and supervised
        # point-sampling RNG. The view index cycles evenly through all four.
        result = model(views[step % 4], [x.cuda() for x in batch["mask_labels"]],
                       [x.cuda() for x in batch["class_labels"]])
        supervised, _ = objective(result, batch, config, pos_weight, station_weight, class_weight)
        if not torch.isfinite(supervised):
            raise RuntimeError("Nonfinite supervised objective")
        supervised.backward()
        sup_value = float(supervised.detach())
        del result, supervised
        ic_value = 0.0
        sup_norm = grad_norm(model) if args.audit else None
        do_ic = probe is not None and probe["pixels"] > 0 and step % args.ic_every == 0
        # Matched augmentation also measures IC, but never backpropagates it.
        if do_ic:
            routing = "geo" if args.arm == "aug" else args.arm
            weights = probe["weights"][routing]
            with torch.no_grad():
                margins = [log_margin(model(view)["fine_probability"], probe, size) for view in views]
                delta = mixed_response(margins)
                ic_value = float((weights * delta.square()).sum())
                coefficient = 2 * weights * delta * args.ic_weight
            if args.arm != "aug":
                # Exact derivative of sum(w*delta^2), using detached delta and
                # deterministic replay: only ONE forward graph lives at a time.
                # No parameter update between the four replays.
                if args.audit:
                    supervised_grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
                    optimizer.zero_grad(set_to_none=True)
                for sign, view in zip((1, -1, -1, 1), views):
                    margin = log_margin(model(view)["fine_probability"], probe, size)
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
            else:
                ic_norm = 0.0
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        if not np.isfinite(norm):
            raise RuntimeError("Nonfinite gradient norm")
        optimizer.step()
        row = {"step": step + 1, "name": item["name"], "supervised": sup_value,
               "ic_raw": ic_value, "gradient_norm": norm,
               "routed_pixels": 0 if probe is None else probe["pixels"],
               "boundary_pixels": 0 if probe is None else probe["boundary_pixels"],
               "seconds": time.monotonic()-start,
               "peak_memory_mb": torch.cuda.max_memory_allocated()/1024**2}
        if args.audit:
            row.update(supervised_grad_norm=sup_norm,
                       weighted_ic_grad_norm=ic_norm if do_ic else 0)
        history.append(row)
        if step % 8 == 0 or step + 1 == args.steps:
            save_json(out / "history.json", history)
            print(json.dumps(row), flush=True)
        if time.monotonic() - start > args.max_seconds:
            save_json(out / "timeout.json", {"completed_steps": step+1})
            raise TimeoutError("Per-arm time budget exhausted")
    if args.audit:
        save_json(out / "completion.json", {"status": "audit_completed", "steps": args.steps,
                                            "seconds": time.monotonic()-start})
        return
    # Fixed final-step checkpoint: no per-arm best-epoch search.
    torch.save({"model": model.cpu().state_dict(), "epoch": args.steps,
                "geosurg_ic": provenance}, out / "best.pt")
    model.cuda().eval()
    loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate)
    print(json.dumps({"event": "validation_started", "frames": len(val_ds)}), flush=True)
    metrics = evaluate(model, loader, torch.device("cuda"), labelmap, config)
    save_json(out / "validation.json", {"grid": [config["height"], config["width"]],
        "split": "validation_used_for_original_checkpoint_selection", "metrics": metrics})
    save_json(out / "completion.json", {"status": "completed", "steps": args.steps,
                                        "seconds": time.monotonic()-start})
    print(json.dumps({"event": "completed", "arm": args.arm,
        "metrics": {k: v for k, v in metrics.items() if isinstance(v, float)}}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["aug", "gt", "geo", "shuffle"], required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--steps", type=int, default=192)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--ic-weight", type=float, default=0.02)
    p.add_argument("--ic-every", type=int, default=2)
    p.add_argument("--max-seconds", type=int, default=5400)
    p.add_argument("--audit", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    assert torch.cuda.device_count() == 1, "Exactly one GPU must be visible"
    torch.set_num_threads(4)
    config = yaml.safe_load((BASE / "configs/d517_task2_fine_p2.yaml").read_text())
    manifest = load_manifest(BASE / "splits/case_folds_d517.json")
    current = build_trainval_manifest(ROOT / "data", int(config["split_seed"]))
    assert current["snapshot"] == manifest["snapshot"], "Dataset changed"
    labelmap = LabelMap.load(ROOT / "data/labelmap.csv")
    config["fine_to_coarse"] = labelmap.fine_to_coarse.astype(int).tolist()
    out = args.output / args.arm / f"fold_{args.fold}"
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    out.mkdir(parents=True)
    train(args, config, manifest, labelmap, out)


if __name__ == "__main__":
    main()
