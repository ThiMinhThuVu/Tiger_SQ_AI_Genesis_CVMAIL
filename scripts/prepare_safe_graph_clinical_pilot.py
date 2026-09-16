#!/usr/bin/env python3
"""Build a compact, image-grounded SAFE-Graph clinical-review pilot package."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[1]
EXPERT_FIELDS = [
    "expert_decision_valid_invalid_depends",
    "expert_relation_scope",
    "expert_constraint_hard_soft",
    "expert_absence_is_violation",
    "expert_warning_eligible",
    "expert_correction_eligible",
    "expert_camera_dependency",
    "expert_tool_occlusion_dependency",
    "expert_exception",
    "reviewer_id",
    "review_date",
    "review_comment",
]
COMPONENT_REVIEW_FIELDS = [
    "clinical_error_valid_yes_no_uncertain",
    "clinical_error_family",
    "clinical_severity_0_1_2_3",
    "warning_would_be_useful_yes_no_uncertain",
    "knowledge_cue",
    "tool_occluded_yes_no_uncertain",
    "camera_ambiguous_yes_no_uncertain",
    "reviewer_id",
    "review_date",
    "review_comment",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--oof-root", type=Path,
        default=ROOT / "artifacts/safe_graph_oof/improved_v2_validation_v1",
    )
    parser.add_argument(
        "--error-root", type=Path,
        default=ROOT / "artifacts/safe_graph_error_corpus/improved_v2_validation_error_v1_r1",
    )
    parser.add_argument(
        "--expert-root", type=Path,
        default=ROOT / "artifacts/safe_graph_knowledge/kg_v1_20260808/expert_review_v1",
    )
    parser.add_argument("--image-root", type=Path, default=ROOT / "data/images")
    parser.add_argument("--labelmap", type=Path, default=ROOT / "data/labelmap.csv")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--top-confusion-pairs", type=int, default=5)
    parser.add_argument("--examples-per-pair", type=int, default=4)
    parser.add_argument("--examples-per-error-family", type=int, default=4)
    parser.add_argument("--complete-miss-examples", type=int, default=10)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fieldnames or list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def area_bin(area_fraction: float) -> str:
    if area_fraction < 0.0005:
        return "tiny"
    if area_fraction < 0.002:
        return "small"
    if area_fraction < 0.01:
        return "medium"
    return "large"


def global_component_mask(mask: np.ndarray, class_id: int, component_id: int) -> np.ndarray:
    """Recover a class-wise component using the frozen corpus enumeration order."""
    offset = 0
    structure = np.ones((3, 3), dtype=np.uint8)
    for current_class in range(1, class_id + 1):
        labels, count = ndimage.label(mask == current_class, structure=structure)
        if current_class == class_id:
            local_id = component_id - offset
            if local_id < 1 or local_id > count:
                raise ValueError(
                    f"Component {component_id} is invalid for class {class_id}; "
                    f"offset={offset}, class_count={count}"
                )
            return labels == local_id
        offset += count
    raise ValueError(f"Class {class_id} not reached")


def diverse_select(rows: list[dict], limit: int) -> list[dict]:
    """Deterministic case/area-bin round-robin, preferring larger components."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        fraction = float(row.get("area_fraction") or 0.0)
        if not fraction and row.get("area_pixels"):
            fraction = float(row["area_pixels"]) / (512 * 896)
        groups[(row["case_id"], area_bin(fraction))].append(row)
    for values in groups.values():
        values.sort(key=lambda item: (-int(float(item["area_pixels"])), item["frame"]))
    selected: list[dict] = []
    keys = sorted(groups)
    while len(selected) < limit:
        progressed = False
        for key in keys:
            if groups[key] and len(selected) < limit:
                selected.append(groups[key].pop(0))
                progressed = True
        if not progressed:
            break
    return selected


def select_priority_tables(
    station_rows: list[dict], relation_rows: list[dict], confusion_rows: list[dict], top_pairs: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    p0_confusions = sorted(
        (row for row in confusion_rows if row["review_priority"] == "P0"),
        key=lambda row: -float(row["edge_weight"]),
    )[:top_pairs]
    selected_classes = {
        int(row[key]) for row in p0_confusions for key in ("class_i", "class_j")
    }
    selected_pairs = {
        frozenset((int(row["class_i"]), int(row["class_j"]))) for row in p0_confusions
    }
    station_priority = [
        row for row in station_rows
        if row["review_priority"] == "P0" or int(row["anatomy_id"]) in selected_classes
    ]
    relation_priority = [
        row for row in relation_rows
        if row["review_priority"] == "P0"
        or frozenset((int(row["subject_id"]), int(row["object_id"]))) in selected_pairs
    ]
    return station_priority, relation_priority, p0_confusions


def select_component_examples(
    predicted_rows: list[dict], gt_rows: list[dict], confusion_rows: list[dict],
    examples_per_pair: int, examples_per_family: int, complete_miss_examples: int,
) -> list[dict]:
    selected: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    for pair in confusion_rows:
        class_i, class_j = int(pair["class_i"]), int(pair["class_j"])
        candidates = [
            row for row in predicted_rows
            if row["error_label"] == "CLASS_SWAP"
            and {int(row["predicted_class_id"]), int(row["dominant_gt_class_id"])}
            == {class_i, class_j}
        ]
        for row in diverse_select(candidates, examples_per_pair):
            row = dict(row)
            row["selection_source"] = "priority_confusion_pair"
            row["linked_rule_id"] = pair["confusion_rule_id"]
            key = ("predicted_component", row["frame"], row["component_id"])
            if key not in seen:
                selected.append(row)
                seen.add(key)

    predicted_families = [
        "CLASS_SWAP", "FRAGMENTED_PREDICTION", "BOUNDARY_OR_PARTIAL_ERROR",
        "BACKGROUND_HALLUCINATION", "ABSENT_CLASS_HALLUCINATION",
        "AMBIGUOUS_FALSE_POSITIVE",
    ]
    for family in predicted_families:
        candidates = [
            row for row in predicted_rows
            if row["error_label"] == family and truthy(row["dangerous_error_candidate"])
        ]
        for row in diverse_select(candidates, examples_per_family):
            key = ("predicted_component", row["frame"], row["component_id"])
            if key in seen:
                continue
            row = dict(row)
            row["selection_source"] = "dangerous_error_family_stratum"
            row["linked_rule_id"] = ""
            selected.append(row)
            seen.add(key)

    misses = [
        row for row in gt_rows
        if row["error_label"] == "COMPLETE_MISS" and truthy(row["dangerous_error_candidate"])
    ]
    for row in diverse_select(misses, complete_miss_examples):
        row = dict(row)
        row["selection_source"] = "dangerous_complete_miss_stratum"
        row["linked_rule_id"] = ""
        row["area_fraction"] = float(row["area_pixels"]) / (512 * 896)
        selected.append(row)
    return selected


def colorize(mask: np.ndarray, colors: np.ndarray) -> np.ndarray:
    return colors[np.clip(mask.astype(np.int64), 0, len(colors) - 1)]


def blend(image: np.ndarray, color_mask: np.ndarray, active: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    output = image.astype(np.float32).copy()
    output[active] = (1 - alpha) * output[active] + alpha * color_mask[active]
    return np.clip(output, 0, 255).astype(np.uint8)


def contour(image: np.ndarray, focus: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    output = image.copy()
    edge = ndimage.binary_dilation(focus, iterations=2) ^ ndimage.binary_erosion(focus, iterations=1)
    output[edge] = color
    return output


def probability_heat(image: np.ndarray, values: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    weight = np.clip(values.astype(np.float32), 0, 1)[..., None]
    tint = np.zeros_like(image, dtype=np.float32)
    tint[:] = color
    return np.clip(image * (1 - 0.7 * weight) + tint * (0.7 * weight), 0, 255).astype(np.uint8)


def crop_bounds(focus: np.ndarray, padding: int = 40) -> tuple[int, int, int, int]:
    ys, xs = np.where(focus)
    if not len(xs):
        return 0, 0, focus.shape[1], focus.shape[0]
    return (
        max(0, int(xs.min()) - padding), max(0, int(ys.min()) - padding),
        min(focus.shape[1], int(xs.max()) + padding + 1),
        min(focus.shape[0], int(ys.max()) + padding + 1),
    )


def panel(array: np.ndarray, title: str, size: tuple[int, int], crop=None) -> Image.Image:
    image = Image.fromarray(array)
    if crop is not None:
        image = image.crop(crop)
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size[0], size[1] + 26), "white")
    canvas.paste(image, ((size[0] - image.width) // 2, 26))
    ImageDraw.Draw(canvas).text((6, 6), title, fill="black")
    return canvas


def render_montage(
    row: dict, arrays: dict[str, np.ndarray], image_path: Path,
    colors: np.ndarray, output_path: Path,
) -> None:
    image = np.asarray(Image.open(image_path).convert("RGB").resize(
        (arrays["prediction"].shape[1], arrays["prediction"].shape[0]), Image.Resampling.BILINEAR
    ))
    prediction = arrays["prediction"]
    target = arrays["target"]
    is_gt = row["evidence_type"] == "gt_component"
    class_id = int(row["focus_class_id"])
    component_id = int(row["component_id"])
    focus = global_component_mask(target if is_gt else prediction, class_id, component_id)
    pred_overlay = blend(image, colorize(prediction, colors), prediction > 0)
    gt_overlay = blend(image, colorize(target, colors), target > 0)
    pred_overlay = contour(pred_overlay, focus, (255, 255, 255)) if not is_gt else pred_overlay
    gt_overlay = contour(gt_overlay, focus, (255, 255, 255)) if is_gt else gt_overlay
    entropy = arrays["entropy"].astype(np.float32) / max(math.log(len(colors)), 1.0)
    uncertainty = probability_heat(image, entropy, (255, 80, 0))
    tool = probability_heat(image, arrays["tool_probability"], (255, 0, 255))
    views = [image, pred_overlay, gt_overlay, uncertainty, tool]
    titles = ["RGB", "Prediction", "Ground truth", "Entropy", "Tool probability"]
    full_panels = [panel(view, title, (448, 256)) for view, title in zip(views, titles)]
    crop = crop_bounds(focus)
    crop_panels = [panel(view, f"Crop: {title}", (448, 256), crop) for view, title in zip(views, titles)]
    header = Image.new("RGB", (2240, 64), "white")
    descriptor = (
        f"{row['review_id']} | {row['case_id']} | {row['frame']} | "
        f"{row['automatic_error_label']} | {row['focus_class_name']} | "
        f"area={row['area_pixels']} | linked={row['linked_rule_id'] or 'none'}"
    )
    ImageDraw.Draw(header).text((8, 8), descriptor, fill="black")
    montage = Image.new("RGB", (2240, 64 + 282 * 2), "white")
    montage.paste(header, (0, 0))
    for index, item in enumerate(full_panels):
        montage.paste(item, (448 * index, 64))
    for index, item in enumerate(crop_panels):
        montage.paste(item, (448 * index, 64 + 282))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    montage.save(output_path, quality=92)


def frame_array_lookup(oof_root: Path) -> dict[str, Path]:
    lookup = {}
    for fold in range(5):
        fold_dir = oof_root / f"fold_{fold}"
        for row in read_csv(fold_dir / "frame_index.csv"):
            lookup[row["frame"]] = fold_dir / row["array_file"]
    return lookup


def to_review_row(row: dict, index: int) -> dict:
    is_gt = "gt_component_id" in row
    class_prefix = "gt" if is_gt else "predicted"
    fraction = float(row.get("area_fraction") or 0.0)
    if not fraction:
        fraction = float(row["area_pixels"]) / (512 * 896)
    base = {
        "review_id": f"CE-{index:03d}",
        "evidence_type": "gt_component" if is_gt else "predicted_component",
        "selection_source": row["selection_source"],
        "linked_rule_id": row.get("linked_rule_id", ""),
        "fold": row["fold"],
        "case_id": row["case_id"],
        "frame": row["frame"],
        "component_id": row["gt_component_id"] if is_gt else row["component_id"],
        "focus_class_id": row[f"{class_prefix}_class_id"],
        "focus_class_name": row[f"{class_prefix}_class_name"],
        "automatic_error_label": row["error_label"],
        "area_pixels": row["area_pixels"],
        "area_fraction": fraction,
        "area_bin": area_bin(fraction),
        "clinical_risk_weight": row["clinical_risk_weight"],
        "predicted_class_name": row.get("predicted_class_name", row.get("dominant_predicted_class_name", "")),
        "dominant_gt_class_name": row.get("dominant_gt_class_name", row.get("gt_class_name", "")),
        "confidence": row.get("mean_predicted_class_probability", ""),
        "margin": row.get("mean_top1_top2_margin", ""),
        "entropy": row.get("mean_entropy", ""),
        "mean_tool_probability": row.get("mean_tool_probability", ""),
        "montage_path": f"montages/CE-{index:03d}.jpg",
    }
    return {**base, **{field: "" for field in COMPONENT_REVIEW_FIELDS}}


def build_package(args: argparse.Namespace) -> dict:
    args.output_root.mkdir(parents=True, exist_ok=True)
    predicted_path = args.error_root / "predicted_component_errors.csv"
    gt_path = args.error_root / "gt_component_errors.csv"
    station_path = args.expert_root / "station_anatomy_review.csv"
    relation_path = args.expert_root / "anatomy_relation_review.csv"
    confusion_path = args.expert_root / "confusion_risk_review.csv"
    predicted_rows, gt_rows = read_csv(predicted_path), read_csv(gt_path)
    station_rows, relation_rows, confusion_rows = (
        read_csv(station_path), read_csv(relation_path), read_csv(confusion_path)
    )
    station_priority, relation_priority, confusion_priority = select_priority_tables(
        station_rows, relation_rows, confusion_rows, args.top_confusion_pairs
    )
    examples = select_component_examples(
        predicted_rows, gt_rows, confusion_priority,
        args.examples_per_pair, args.examples_per_error_family, args.complete_miss_examples,
    )
    review_rows = [to_review_row(row, index) for index, row in enumerate(examples, 1)]
    write_csv(args.output_root / "component_error_review.csv", review_rows)
    write_csv(args.output_root / "priority_station_review.csv", station_priority)
    write_csv(args.output_root / "priority_anatomy_relation_review.csv", relation_priority)
    write_csv(args.output_root / "priority_confusion_review.csv", confusion_priority)

    label_rows = read_csv(args.labelmap)
    colors = np.asarray([
        [int(row["fine_r"]), int(row["fine_g"]), int(row["fine_b"])] for row in label_rows
    ], dtype=np.uint8)
    arrays_by_frame = frame_array_lookup(args.oof_root)
    for row in review_rows:
        arrays = np.load(arrays_by_frame[row["frame"]])
        render_montage(
            row, arrays, args.image_root / row["frame"], colors,
            args.output_root / row["montage_path"],
        )

    source_paths = [predicted_path, gt_path, station_path, relation_path, confusion_path]
    manifest = {
        "artifact": "SAFE-Graph image-grounded clinical review pilot",
        "version": "clinical_pilot_v1",
        "status": "PENDING_EXPERT_REVIEW",
        "split": "out_of_fold_validation_only",
        "test_data_used": False,
        "selection_policy": {
            "top_confusion_pairs": args.top_confusion_pairs,
            "examples_per_pair": args.examples_per_pair,
            "examples_per_error_family": args.examples_per_error_family,
            "complete_miss_examples": args.complete_miss_examples,
            "sampling": "deterministic case/area-bin round-robin",
        },
        "counts": {
            "component_examples": len(review_rows),
            "priority_station_rows": len(station_priority),
            "priority_relation_rows": len(relation_priority),
            "priority_confusion_rows": len(confusion_priority),
            "montages": len(review_rows),
        },
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in source_paths},
        "runtime_authority": "none; expert completion does not enable correction",
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    readme = f"""# SAFE-Graph clinical pilot v1

## Material Passport

- Status: `PENDING_EXPERT_REVIEW`
- Split: `out_of_fold_validation_only`
- Test data used: `false`
- Runtime/correction authority: none

## Review order

1. Open `index.html` and inspect all {len(review_rows)} component montages.
2. Record component decisions in `component_error_review.csv`.
3. Review the {len(confusion_priority)} candidate pairs in `priority_confusion_review.csv`.
4. Review only the related relations in `priority_anatomy_relation_review.csv`.
5. Review station context in `priority_station_review.csv`; absence is not impossibility.
6. Run `scripts/validate_safe_graph_clinical_review.py` after editing.

## Component decision rules

- `clinical_error_valid_yes_no_uncertain`: YES/NO/UNCERTAIN.
- `clinical_severity_0_1_2_3`: 0 none, 1 low, 2 moderate, 3 high.
- Use UNCERTAIN when camera, exposure or tool occlusion prevents judgment.
- `knowledge_cue` should state what anatomy/relation would help verify the error.
- Do not use the automatic taxonomy as the expert answer.

## Safety guardrails

- Empty review fields are intentional.
- Review completion does not promote rules or enable correction.
- Do not edit or inspect held-out test predictions during this review.
- The directory `clinical_pilot_v1` is an invalid precheck artifact; use this directory.
"""
    (args.output_root / "README.md").write_text(readme)
    gallery_items = []
    for row in review_rows:
        gallery_items.append(
            f'<section><h2>{row["review_id"]}: {row["automatic_error_label"]}</h2>'
            f'<p>{row["case_id"]} / {row["frame"]} — {row["focus_class_name"]}; '
            f'area={row["area_bin"]}; linked={row["linked_rule_id"] or "none"}</p>'
            f'<a href="{row["montage_path"]}"><img loading="lazy" src="{row["montage_path"]}"></a>'
            '</section>'
        )
    gallery = """<!doctype html><html><head><meta charset="utf-8">
<title>SAFE-Graph clinical pilot</title><style>
body{font-family:sans-serif;max-width:1500px;margin:auto;padding:20px}
section{border-bottom:1px solid #bbb;padding:12px 0}img{width:100%;height:auto}
</style></head><body><h1>SAFE-Graph clinical pilot v1</h1>
<p>Images are OOF validation evidence. Record decisions in component_error_review.csv.</p>
""" + "\n".join(gallery_items) + "</body></html>\n"
    (args.output_root / "index.html").write_text(gallery)
    return manifest


def main() -> None:
    manifest = build_package(parse_args())
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
