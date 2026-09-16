from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

from tiger_models.metrics import binary_dice, normalized_hausdorff, official_auroc


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "external/tigersqai_evaluation"


def import_official():
    if str(OFFICIAL) not in sys.path:
        sys.path.insert(0, str(OFFICIAL))
    from metrics.metrics import _binary_dice, _binary_normalized_hausdorff, auroc
    return _binary_dice, _binary_normalized_hausdorff, auroc


def test_fast_metrics_conform_to_official_primitives() -> None:
    official_dice, official_hd, official_auc = import_official()
    prediction = np.zeros((18, 30), dtype=bool)
    target = np.zeros_like(prediction)
    prediction[2:8, 4:12] = True
    target[4:11, 7:15] = True
    diagonal = float(np.hypot(*prediction.shape))
    assert binary_dice(prediction, target) == official_dice(prediction, target)
    assert np.isclose(normalized_hausdorff(prediction, target), official_hd(prediction, target, diagonal))
    truth = np.asarray([0, 1, 0, 1, 1, 0])
    probability = np.asarray([0.1, 0.8, 0.4, 0.9, 0.7, 0.2])
    assert official_auroc(truth, probability) == official_auc(truth, probability)


def test_official_evaluator_accepts_generated_rgb_prediction(tmp_path: Path) -> None:
    if str(OFFICIAL) not in sys.path:
        sys.path.insert(0, str(OFFICIAL))
    from metrics.evaluate_seg import evaluate
    gt, pred = tmp_path / "gt", tmp_path / "pred"
    gt.mkdir(); pred.mkdir()
    mask = np.zeros((12, 20, 3), dtype=np.uint8)
    mask[2:8, 4:11] = (255, 0, 204)
    Image.fromarray(mask, mode="RGB").save(gt / "center_1_case_10_9.png")
    Image.fromarray(mask, mode="RGB").save(pred / "center_1_case_10_9.png")
    result = evaluate(gt, pred)
    assert result["final_dice"] == 1.0
    assert result["final_hd"] == 0.0
