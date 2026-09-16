#!/usr/bin/env python3
"""Build dense-watershed CholecSeg8k manifests aligned to the SP-TCN test split."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

LABEL_IDS = [50, 11, 21, 13, 12, 31, 23, 24, 25, 32, 22, 33, 5]
TEST_VIDEOS = {"video12", "video20", "video48", "video55"}


def balanced_folds(counts: dict[str, int], n_folds: int, seed: int) -> list[list[str]]:
    rng = random.Random(seed)
    tie = {video: rng.random() for video in counts}
    ordered = sorted(counts, key=lambda video: (-counts[video], tie[video]))
    folds, totals = [[] for _ in range(n_folds)], [0] * n_folds
    for video in ordered:
        fold = min(range(n_folds), key=lambda i: (totals[i], len(folds[i]), i))
        folds[fold].append(video); totals[fold] += counts[video]
    return [sorted(fold) for fold in folds]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--split-dir", default="sptcn_split_dense_seed2026")
    args = parser.parse_args()
    root = args.data_root.resolve(); out = root / args.split_dir
    masks = out / "semantic_masks"; masks.mkdir(parents=True, exist_ok=True)
    images = sorted(root.glob("video*/*/*_endo.png"))
    videos = {path.parts[-3] for path in images}
    missing = TEST_VIDEOS - videos
    if missing: raise RuntimeError(f"SP-TCN test videos missing: {sorted(missing)}")
    lut = np.full(256, 255, dtype=np.uint8)
    for contiguous, source in enumerate(LABEL_IDS): lut[source] = contiguous
    lut[0] = 0; lut[255] = 0  # black/white image-border variants
    lut[35] = 2   # alternate liver ID
    rows=[]; counts={}
    for i, image_path in enumerate(images):
        rel=image_path.relative_to(root); video=rel.parts[0]
        counts[video]=counts.get(video,0)+1
        source=image_path.with_name(image_path.name.replace("_endo.png","_endo_watershed_mask.png"))
        raw=np.asarray(Image.open(source))[...,0]
        converted=lut[raw]
        unknown=np.unique(raw[converted==255])
        if len(unknown): raise ValueError(f"{source}: unknown IDs {unknown.tolist()}")
        target=masks/f"{video}_{rel.parts[1]}_{image_path.stem}.png"
        if not target.exists(): Image.fromarray(converted).save(target)
        rows.append({"video":video,"image":rel.as_posix(),"mask":target.relative_to(root).as_posix()})
        if (i+1)%1000==0: print(f"prepared {i+1}/{len(images)}",flush=True)
    dev=set(counts)-TEST_VIDEOS
    folds=balanced_folds({v:counts[v] for v in dev},5,args.seed)
    def write(name: str, selected: set[str]) -> int:
        payload=[{"image":r["image"],"mask":r["mask"]} for r in rows if r["video"] in selected]
        (out/name).write_text(json.dumps(payload,indent=2)+"\n"); return len(payload)
    test_n=write("test.json",TEST_VIDEOS)
    fold_meta=[]
    for fold,val_list in enumerate(folds):
        val=set(val_list); train=dev-val
        fold_meta.append({"fold":fold,"train_videos":sorted(train),"validation_videos":sorted(val),
                          "train_frames":write(f"fold{fold}_train.json",train),
                          "validation_frames":write(f"fold{fold}_val.json",val)})
    metadata={"protocol":"SP-TCN fixed test videos; 5-fold video-level validation on remaining videos",
              "reference":"https://doi.org/10.1007/s11548-023-02971-6","seed":args.seed,
              "mask_source":"*_endo_watershed_mask.png","total_frames":len(rows),
              "test_videos":sorted(TEST_VIDEOS),"test_frames":test_n,"folds":fold_meta}
    (out/"splits.json").write_text(json.dumps(metadata,indent=2)+"\n")
    digest=hashlib.sha256((out/"splits.json").read_bytes()).hexdigest()
    print(json.dumps({"output":str(out),"split_sha256":digest,**metadata},indent=2))


if __name__ == "__main__": main()
