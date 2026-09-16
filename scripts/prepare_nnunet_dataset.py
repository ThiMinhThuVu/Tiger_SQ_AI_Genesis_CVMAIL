#!/usr/bin/env python3
"""Convert TIGER PNG frames to a case-safe nnU-Net v2 2D dataset."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tiger_models.data import TEST_CASES, VALIDATION_CASES, LabelMap, case_from_name, records_for_fold


def write_nifti(array: np.ndarray, path: Path) -> None:
    image = sitk.GetImageFromArray(array.astype(np.uint8)[None, ...])
    image.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(image, str(path))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=ROOT / "data")
    p.add_argument("--output-root", type=Path, default=ROOT / "artifacts/nnunet_raw")
    p.add_argument("--dataset-id", type=int, default=501)
    args = p.parse_args()
    labelmap = LabelMap.load(args.data_root / "labelmap.csv")
    dataset = args.output_root / f"Dataset{args.dataset_id:03d}_TIGER"
    images, labels = dataset / "imagesTr", dataset / "labelsTr"
    images.mkdir(parents=True, exist_ok=True); labels.mkdir(parents=True, exist_ok=True)
    frame_names = sorted(p.name for p in (args.data_root / "images").glob("*.png"))
    for name in frame_names:
        stem = Path(name).stem
        image = np.asarray(Image.open(args.data_root / "images" / name).convert("L"))
        fine_rgb = np.asarray(Image.open(args.data_root / "masks_fine" / name).convert("RGB"))
        fine = labelmap.decode_fine(fine_rgb)
        write_nifti(image, images / f"{stem}_0000.nii.gz")
        write_nifti(fine, labels / f"{stem}.nii.gz")
    (dataset / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "grayscale"},
        "labels": {"background": 0, **{labelmap.fine_names[i]: i for i in range(1, 31)}},
        "numTraining": len(frame_names), "file_ending": ".nii.gz", "name": "TIGER",
    }, indent=2) + "\n")
    splits = []
    for fold in range(5):
        train = records_for_fold(args.data_root, fold, "train")
        val = records_for_fold(args.data_root, fold, "validation")
        splits.append({"train": [Path(r.name).stem for r in train],
                       "val": [Path(r.name).stem for r in val]})
    (dataset / "splits_final.json").write_text(json.dumps(splits, indent=2) + "\n")
    print(json.dumps({"dataset": str(dataset), "frames": len(frame_names), "folds": 5}, indent=2))


if __name__ == "__main__":
    main()
