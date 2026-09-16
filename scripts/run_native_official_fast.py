#!/usr/bin/env python3
"""Run official evaluation with an exact, scalable Hausdorff primitive."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--figures-dir", default="none")
    parser.add_argument("--official-evaluator", type=Path, default=Path("external/tigersqai_evaluation"))
    args = parser.parse_args()
    evaluator = args.official_evaluator.resolve()
    if not (evaluator / "metrics/01_evaluate_challenge.py").is_file():
        raise FileNotFoundError(evaluator)

    old = """    pred_pts = np.argwhere(pred.astype(bool))
    gt_pts   = np.argwhere(gt.astype(bool))

    if len(pred_pts) == 0 and len(gt_pts) == 0:
        return 0.0
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return 1.0

    hd = max(
        _directed_hausdorff(pred_pts, gt_pts),
        _directed_hausdorff(gt_pts,   pred_pts),
    )
    return float(min(hd / diagonal, 1.0))
"""
    new = """    pred_mask = pred.astype(bool)
    gt_mask = gt.astype(bool)
    if not pred_mask.any() and not gt_mask.any():
        return 0.0
    if not pred_mask.any() or not gt_mask.any():
        return 1.0

    # Exact equivalent of max_a min_b ||a-b|| on the binary pixel grid.
    from scipy.ndimage import distance_transform_edt
    pred_to_gt = float(distance_transform_edt(~gt_mask)[pred_mask].max(initial=0.0))
    gt_to_pred = float(distance_transform_edt(~pred_mask)[gt_mask].max(initial=0.0))
    hd = max(pred_to_gt, gt_to_pred)
    return float(min(hd / diagonal, 1.0))
"""

    with tempfile.TemporaryDirectory(prefix="tigersqai-official-fast-") as temp:
        work = Path(temp) / "evaluator"
        shutil.copytree(evaluator, work)
        metrics_file = work / "metrics/metrics.py"
        source = metrics_file.read_text()
        if old not in source:
            raise RuntimeError("official evaluator metric primitive did not match expected source")
        metrics_file.write_text(source.replace(old, new, 1))
        command = [
            sys.executable, str(work / "metrics/01_evaluate_challenge.py"),
            "--gt", str(args.gt.resolve()), "--pred", str(args.pred.resolve()),
            "--out", str(args.out.resolve()), "--figures-dir", args.figures_dir,
            "--save-json",
        ]
        process = subprocess.run(command, cwd=work, text=True, capture_output=True)
        args.out.with_suffix(".log").write_text(process.stdout + process.stderr)
        print(process.stdout, end="")
        print(f"fast_official_returncode={process.returncode}")
        return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
