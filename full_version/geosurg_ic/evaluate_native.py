#!/usr/bin/env python3
"""Paired native-resolution validation, with GT-boundary-band diagnostics."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from full_version.geosurg_ic.run_pilot import BASE, save_json
from full_version.task2_fine_31cls.scripts.data_full import load_manifest, records_for_manifest
from scripts.train_mask2former_improved import build_improved
from tiger_models.data import LabelMap, TigerMultitaskDataset
from tiger_models.improved_mask2former import soft_presence_modulation
from tiger_models.metrics import binary_dice, normalized_hausdorff


def score_frame(record, prediction, labelmap):
    with Image.open(record.fine_mask) as im:
        target = labelmap.decode_fine(np.asarray(im.convert("RGB")))
    assert prediction.shape == target.shape
    boundary = np.zeros(target.shape, dtype=bool)
    horizontal = target[:, 1:] != target[:, :-1]
    vertical = target[1:] != target[:-1]
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    boundary[1:] |= vertical
    boundary[:-1] |= vertical
    # Resolution-relative GT band; not the standard Boundary IoU metric.
    radius = max(1, round(.003 * np.hypot(*target.shape)))
    band = binary_dilation(boundary, iterations=radius)
    dices, nhds, boundary_dices, present_weights = [], [], [], []
    for c, weight in enumerate(labelmap.fine_weights):
        pred, gt = prediction == c, target == c
        dices.append(binary_dice(pred, gt))
        nhds.append(normalized_hausdorff(pred, gt))
        if (gt & band).any() and weight > 0:
            boundary_dices.append(binary_dice(pred & band, gt & band))
            present_weights.append(weight)
    weights = labelmap.fine_weights
    d, n = float(np.average(dices, weights=weights)), float(np.average(nhds, weights=weights))
    return {"name": record.name, "case": record.case_id,
            "dice": d, "nhd": n, "score": (d+1-n)/2,
            "boundary_band_dice": float(np.average(boundary_dices, weights=present_weights)) if present_weights else None,
            "boundary_band_accuracy": float((prediction[band] == target[band]).mean()) if band.any() else None,
            "band_radius_px": radius, "native_shape": list(target.shape)}


def aggregate(rows):
    keys = ("dice", "nhd", "score", "boundary_band_dice", "boundary_band_accuracy")
    cases = {}
    for case in sorted({r["case"] for r in rows}):
        selected = [r for r in rows if r["case"] == case]
        cases[case] = {k: float(np.mean([r[k] for r in selected if r[k] is not None])) for k in keys}
    centers = {}
    for center in sorted({case.split("_case_")[0] for case in cases}):
        selected = [v for case, v in cases.items() if case.split("_case_")[0] == center]
        centers[center] = {k: float(np.mean([r[k] for r in selected])) for k in keys}
    return {"per_case": cases, "per_center": centers,
            "case_macro": {k: float(np.mean([r[k] for r in cases.values()])) for k in keys},
            "center_macro": {k: float(np.mean([r[k] for r in centers.values()])) for k in keys},
            "worst_center_score": min(r["score"] for r in centers.values())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pilot", type=Path, required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--arms", nargs="+", default=["baseline", "aug", "geo", "gt", "shuffle"])
    args = p.parse_args()
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(4)
    config = yaml.safe_load((BASE / "configs/d517_task2_fine_p2.yaml").read_text())
    manifest = load_manifest(BASE / "splits/case_folds_d517.json")
    records = records_for_manifest(ROOT / "data", manifest, args.fold, "validation")
    labelmap = LabelMap.load(ROOT / "data/labelmap.csv")
    ds = TigerMultitaskDataset(records, labelmap, config["height"], config["width"], False)
    results = {}
    start = time.monotonic()
    for arm in args.arms:
        out = args.pilot / "native" / f"fold_{args.fold}" / (arm + ".json")
        if out.exists():
            raise FileExistsError(out)
        checkpoint = (BASE / "artifacts" / config["experiment_id"] / f"fold_{args.fold}/best.pt"
                      if arm == "baseline" else args.pilot / arm / f"fold_{args.fold}/best.pt")
        model = build_improved(config, labelmap)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        del state
        model.cuda().eval()
        rows, pending = [], []
        with ThreadPoolExecutor(max_workers=4) as pool:
            for i, record in enumerate(records):
                item = ds[i]
                with torch.inference_mode():
                    result = model(item["image"][None].cuda())
                    native_size = tuple(int(x) for x in item["original_size"])
                    fine = F.interpolate(result["fine_probability"], native_size, mode="bilinear", align_corners=False)
                    fine = soft_presence_modulation(fine, result["presence_logits"], config["presence_epsilon"], config["presence_gamma"])
                    prediction = fine.argmax(1)[0].byte().cpu().numpy()
                    del result, fine
                pending.append(pool.submit(score_frame, record, prediction, labelmap))
                if len(pending) >= 4:
                    rows.append(pending.pop(0).result())
                if i % 16 == 0:
                    print(json.dumps({"arm": arm, "inferred": i+1, "total": len(records),
                                      "seconds": time.monotonic()-start}), flush=True)
            rows.extend(f.result() for f in pending)
        del model
        torch.cuda.empty_cache()
        results[arm] = {"checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "split": "D517 validation, previously used for checkpoint selection",
            "grid": "native, probabilities resized before argmax", "frames": len(rows),
            "rows": rows, **aggregate(rows)}
        save_json(out, results[arm])
        print(json.dumps({"arm": arm, "complete": True,
                          "center_macro": results[arm]["center_macro"]}), flush=True)
    save_json(args.pilot / "native" / f"fold_{args.fold}" / "comparison.json", results)


if __name__ == "__main__":
    main()
