#!/usr/bin/env python3
"""Infer each nnU-Net fold on its two held-out test cases and summarize metrics."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import SimpleITK as sitk
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tiger_models.data import records_for_fold


def write_input(record, path: Path) -> None:
    image = np.asarray(Image.open(record.image).convert("L"), dtype=np.uint8)
    nifti = sitk.GetImageFromArray(image[None, ...])
    nifti.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(nifti, str(path))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=ROOT / "data")
    p.add_argument("--results-root", type=Path, default=ROOT / "artifacts/nnunet_results/Dataset501_TIGER/nnUNetTrainer_TIGER100__nnUNetPlans__2d")
    p.add_argument("--work-root", type=Path, default=ROOT / "artifacts/nnunet_test")
    p.add_argument("--dataset-id", type=int, default=501)
    p.add_argument("--trainer", default="nnUNetTrainer_TIGER100")
    p.add_argument("--config", default="2d")
    args = p.parse_args()
    args.work_root.mkdir(parents=True, exist_ok=True)
    for fold in range(5):
        input_dir = args.work_root / f"input/fold_{fold}"
        output_dir = args.results_root / f"fold_{fold}/test_predictions"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        for record in records_for_fold(args.data_root, fold, "test"):
            write_input(record, input_dir / f"{Path(record.name).stem}_0000.nii.gz")
        checkpoint = args.results_root / f"fold_{fold}/checkpoint_best.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        env = os.environ.copy()
        env.update({
            "nnUNet_raw": str(ROOT / "artifacts/nnunet_raw"),
            "nnUNet_preprocessed": str(ROOT / "artifacts/nnunet_preprocessed"),
            "nnUNet_results": str(ROOT / "artifacts/nnunet_results"),
            "nnUNet_extTrainer": str(ROOT / "jobs/nnunet_external_trainer"),
        })
        command = [
            "nnUNetv2_predict", "-i", str(input_dir), "-o", str(output_dir),
            "-d", str(args.dataset_id), "-c", args.config, "-f", str(fold),
            "-tr", args.trainer, "-chk", "checkpoint_best.pth",
        ]
        subprocess.run(command, cwd=ROOT, env=env, check=True)

    evaluate = [
        "python", str(ROOT / "scripts/evaluate_nnunet_test.py"),
        "--data-root", str(args.data_root),
        "--prediction-root", str(args.results_root),
        "--output", str(args.results_root / "test_quantitative.json"),
    ]
    subprocess.run(evaluate, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
