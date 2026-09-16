from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
from scipy.ndimage import distance_transform_edt


def binary_dice(prediction: np.ndarray, target: np.ndarray) -> float:
    pred_sum = int(prediction.sum())
    target_sum = int(target.sum())
    if pred_sum == 0 and target_sum == 0:
        return 1.0
    if pred_sum == 0 or target_sum == 0:
        return 0.0
    return float(2 * np.logical_and(prediction, target).sum() / (pred_sum + target_sum))


def normalized_hausdorff(prediction: np.ndarray, target: np.ndarray) -> float:
    pred_any, target_any = bool(prediction.any()), bool(target.any())
    if not pred_any and not target_any:
        return 0.0
    if pred_any != target_any:
        return 1.0
    diagonal = float(np.hypot(*prediction.shape))
    # EDT at foreground coordinates gives exactly max_a min_b ||a-b|| used by
    # the official evaluator, but avoids its quadratic point-pair expansion.
    pred_to_target = float(distance_transform_edt(~target)[prediction].max(initial=0.0))
    target_to_pred = float(distance_transform_edt(~prediction)[target].max(initial=0.0))
    return min(max(pred_to_target, target_to_pred) / diagonal, 1.0)


def segmentation_metrics(
    predictions: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    case_ids: Sequence[str],
    class_weights: Sequence[float],
    class_names: Sequence[str],
) -> dict[str, object]:
    if not (len(predictions) == len(targets) == len(case_ids)):
        raise ValueError("Prediction/target/case lengths differ")
    weights = np.asarray(class_weights, dtype=np.float64)
    case_dice: dict[str, list[float]] = defaultdict(list)
    case_nhd: dict[str, list[float]] = defaultdict(list)
    per_class_dice: list[list[float]] = [[] for _ in range(len(weights))]
    per_class_nhd: list[list[float]] = [[] for _ in range(len(weights))]
    observable = np.zeros(len(weights), dtype=bool)
    per_image: list[dict[str, float | str]] = []
    for prediction, target, case_id in zip(predictions, targets, case_ids):
        dices = np.empty(len(weights), dtype=np.float64)
        nhds = np.empty(len(weights), dtype=np.float64)
        for class_id in range(len(weights)):
            pred_binary = prediction == class_id
            target_binary = target == class_id
            observable[class_id] |= bool(target_binary.any())
            dices[class_id] = binary_dice(pred_binary, target_binary)
            nhds[class_id] = normalized_hausdorff(pred_binary, target_binary)
            per_class_dice[class_id].append(float(dices[class_id]))
            per_class_nhd[class_id].append(float(nhds[class_id]))
        weighted_dice = float(np.dot(dices, weights) / weights.sum())
        weighted_nhd = float(np.dot(nhds, weights) / weights.sum())
        case_dice[case_id].append(weighted_dice)
        case_nhd[case_id].append(weighted_nhd)
        per_image.append({"case_id": case_id, "dice": weighted_dice, "nhd": weighted_nhd})
    case_dice_mean = {key: float(np.mean(value)) for key, value in case_dice.items()}
    case_nhd_mean = {key: float(np.mean(value)) for key, value in case_nhd.items()}
    return {
        "dice": float(np.mean(list(case_dice_mean.values()))),
        "nhd": float(np.mean(list(case_nhd_mean.values()))),
        "per_case_dice": case_dice_mean,
        "per_case_nhd": case_nhd_mean,
        "per_image": per_image,
        "per_class": [
            {
                "class_id": index,
                "class_name": class_names[index],
                "weight": float(weights[index]),
                "observable": bool(observable[index]),
                "dice": float(np.mean(per_class_dice[index])) if observable[index] else None,
                "nhd": float(np.mean(per_class_nhd[index])) if observable[index] else None,
                "selection_dice_with_official_absent_convention": float(np.mean(per_class_dice[index])),
                "selection_nhd_with_official_absent_convention": float(np.mean(per_class_nhd[index])),
            }
            for index in range(len(weights))
        ],
    }


def segmentation_failure_metrics(
    predictions: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    class_names: Sequence[str],
) -> dict[str, object]:
    """Rare-class diagnostics whose denominators are explicit and auditable."""
    if len(predictions) != len(targets):
        raise ValueError("Prediction/target lengths differ")
    per_class = []
    confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    for prediction, target in zip(predictions, targets):
        np.add.at(confusion, (target.ravel(), prediction.ravel()), 1)
    for class_id, class_name in enumerate(class_names):
        present_dice, present_images, misses, absent_images, absent_fp = [], 0, 0, 0, 0
        for prediction, target in zip(predictions, targets):
            pred_binary, target_binary = prediction == class_id, target == class_id
            if target_binary.any():
                present_images += 1
                misses += int(not pred_binary.any())
                present_dice.append(binary_dice(pred_binary, target_binary))
            else:
                absent_images += 1
                absent_fp += int(pred_binary.any())
        per_class.append({
            "class_id": class_id, "class_name": class_name,
            "present_images": present_images, "absent_images": absent_images,
            "present_only_dice": float(np.mean(present_dice)) if present_dice else None,
            "complete_miss_rate": misses / present_images if present_images else None,
            "absent_fp_rate": absent_fp / absent_images if absent_images else None,
        })
    return {"per_class": per_class, "confusion_pixels": confusion.tolist()}


def binary_f1(y_true: np.ndarray, y_prediction: np.ndarray) -> float:
    true_positive = int(np.logical_and(y_true == 1, y_prediction == 1).sum())
    false_positive = int(np.logical_and(y_true == 0, y_prediction == 1).sum())
    false_negative = int(np.logical_and(y_true == 1, y_prediction == 0).sum())
    denominator = 2 * true_positive + false_positive + false_negative
    return float(2 * true_positive / denominator) if denominator else 0.0


def official_auroc(y_true: np.ndarray, y_probability: np.ndarray) -> float:
    y_true = y_true.astype(int)
    positives = int(y_true.sum())
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC is undefined for a single-class target")
    order = np.lexsort((-y_true, -y_probability))
    sorted_target = y_true[order]
    tpr = np.concatenate([[0.0], np.cumsum(sorted_target) / positives])
    fpr = np.concatenate([[0.0], np.cumsum(1 - sorted_target) / negatives])
    return float((np.diff(fpr) * (tpr[:-1] + tpr[1:]) / 2.0).sum())


def visibility_metrics(
    target: np.ndarray,
    probability: np.ndarray,
    station_names: Sequence[str],
    threshold: float = 0.5,
) -> dict[str, object]:
    if target.shape != probability.shape or target.shape[1] != len(station_names):
        raise ValueError("Visibility target/probability shape mismatch")
    prediction = (probability >= threshold).astype(int)
    stations = []
    f1_values: list[float] = []
    auroc_values: list[float] = []
    for index, name in enumerate(station_names):
        truth, pred, prob = target[:, index].astype(int), prediction[:, index], probability[:, index]
        f1 = binary_f1(truth, pred)
        tp = int(np.logical_and(truth == 1, pred == 1).sum())
        fp = int(np.logical_and(truth == 0, pred == 1).sum())
        fn = int(np.logical_and(truth == 1, pred == 0).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        try:
            auroc = official_auroc(truth, prob)
            auroc_values.append(auroc)
        except ValueError:
            auroc = None
        f1_values.append(f1)
        stations.append({
            "station": name, "f1": f1, "precision": precision, "recall": recall,
            "auroc": auroc, "observable_positive_and_negative": auroc is not None,
        })
    return {
        "macro_f1": float(np.mean(f1_values)),
        "macro_auroc": float(np.mean(auroc_values)) if auroc_values else None,
        "threshold": threshold,
        "per_station": stations,
    }


def selection_score(
    fine_dice: float,
    fine_nhd: float,
    coarse_dice: float,
    coarse_nhd: float,
    visibility_f1: float,
    visibility_auroc: float | None,
) -> float | None:
    if visibility_auroc is None:
        return None
    values = np.asarray([
        fine_dice, 1.0 - fine_nhd, coarse_dice, 1.0 - coarse_nhd,
        visibility_f1, visibility_auroc,
    ])
    return float(values.mean()) if np.isfinite(values).all() else None
