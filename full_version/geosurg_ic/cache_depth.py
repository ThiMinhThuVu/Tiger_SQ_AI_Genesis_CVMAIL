#!/usr/bin/env python3
"""Offline local pseudo-depth cache. Single visible GPU; never uploads images."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from full_version.task2_fine_31cls.scripts.data_full import load_manifest, records_for_manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--all-folds", action="store_true")
    args = p.parse_args()
    assert torch.cuda.device_count() == 1, "Exactly one GPU must be visible"
    torch.set_num_threads(4)
    manifest = load_manifest(ROOT / "full_version/center_stratified_32_8/splits/case_folds_d517.json")
    records = {}
    for fold in (range(5) if args.all_folds else [args.fold]):
        for r in records_for_manifest(ROOT / "data", manifest, fold, "train"):
            records[r.name] = r
    model_id = "depth-anything/Depth-Anything-V2-Small-hf"
    processor = AutoImageProcessor.from_pretrained(model_id, local_files_only=True)
    model = AutoModelForDepthEstimation.from_pretrained(model_id, local_files_only=True).cuda().eval()
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    rows = []
    for i, r in enumerate(sorted(records.values(), key=lambda r: r.name)):
        path = args.output / (r.name + ".npz")
        digest = hashlib.sha256(r.image.read_bytes()).hexdigest()
        if path.exists():
            with np.load(path) as old:
                assert str(old["sha256"]) == digest
        else:
            with Image.open(r.image) as im:
                inputs = processor(images=im.convert("RGB"), return_tensors="pt").to("cuda")
            with torch.inference_mode():
                prediction = model(**inputs).predicted_depth
                depth = F.interpolate(prediction[:, None].float(), (160, 280), mode="bicubic", align_corners=False)[0, 0]
            if not torch.isfinite(depth).all():
                raise RuntimeError(f"Nonfinite depth: {r.name}")
            np.savez_compressed(path, depth=depth.cpu().numpy().astype(np.float16), sha256=digest)
        rows.append({"name": r.name, "image_sha256": digest})
        if i % 20 == 0:
            print(json.dumps({"cached": i+1, "total": len(records), "seconds": time.monotonic()-start}), flush=True)
    (args.output / "provenance.json").write_text(json.dumps({"model": model_id,
        "revision": model.config._commit_hash, "grid": [160, 280], "records": rows,
        "seconds": time.monotonic()-start, "use": "training-only geometry routing"}, indent=2))
    print("DEPTH_CACHE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
