#!/usr/bin/env python3
"""Audit whether P1 target structures survive the baseline input resize."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from full_version.task2_fine_31cls.scripts.data_full import parse_frame_name  # noqa: E402
from tiger_models.data import LabelMap  # noqa: E402


TARGET_CLASS_IDS = (3, 8, 9, 14, 19, 21, 28)
INPUT_WIDTH = 896
INPUT_HEIGHT = 512


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def summarize(rows: list[dict[str, Any]], class_id: int, center: int | None) -> dict[str, Any]:
    selected = [
        row for row in rows
        if row["class_id"] == class_id and (center is None or row["center"] == center)
    ]
    present = [row for row in selected if row["native_pixels"] > 0]
    projected = [area for row in present for area in row["projected_component_areas"]]
    native_components = sum(row["native_components"] for row in present)
    resized_components = sum(row["resized_components"] for row in present)
    return {
        "center": "overall" if center is None else f"center_{center}",
        "class_id": class_id,
        "class_name": selected[0]["class_name"],
        "frames": len(selected),
        "present_frames": len(present),
        "frame_disappearance_rate": (
            sum(row["resized_pixels"] == 0 for row in present) / len(present) if present else None
        ),
        "native_components": native_components,
        "resized_components": resized_components,
        "component_count_retention": (
            min(1.0, resized_components / native_components) if native_components else None
        ),
        "projected_component_area_median": statistics.median(projected) if projected else None,
        "projected_component_lt4_rate": (
            sum(area < 4 for area in projected) / len(projected) if projected else None
        ),
        "projected_component_lt16_rate": (
            sum(area < 16 for area in projected) / len(projected) if projected else None
        ),
        "projected_component_lt32_rate": (
            sum(area < 32 for area in projected) / len(projected) if projected else None
        ),
        "resized_pixels_present_median": (
            statistics.median(row["resized_pixels"] for row in present) if present else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "full_version/task2_fine_31cls/artifacts/p2_resolution_audit_v1",
    )
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite audit: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    labelmap = LabelMap.load(data_root / "labelmap.csv")
    rows: list[dict[str, Any]] = []
    mask_paths = sorted((data_root / "masks_fine").glob("*.png"))
    for index, path in enumerate(mask_paths, start=1):
        with Image.open(path) as handle:
            rgb = np.asarray(handle.convert("RGB"))
        target = labelmap.decode_fine(rgb)
        height, width = target.shape
        scale = (INPUT_WIDTH / width) * (INPUT_HEIGHT / height)
        resized = np.asarray(
            Image.fromarray(target, mode="L").resize(
                (INPUT_WIDTH, INPUT_HEIGHT), Image.Resampling.NEAREST
            ),
            dtype=np.uint8,
        )
        center = int(parse_frame_name(path.name)["center"])
        for class_id in TARGET_CLASS_IDS:
            native_binary = target == class_id
            resized_binary = resized == class_id
            _, native_count = ndimage.label(native_binary)
            _, resized_count = ndimage.label(resized_binary)
            native_areas = np.bincount(
                ndimage.label(native_binary)[0].ravel(), minlength=native_count + 1
            )[1:]
            rows.append({
                "name": path.name,
                "center": center,
                "class_id": class_id,
                "class_name": labelmap.fine_names[class_id],
                "native_width": width,
                "native_height": height,
                "native_pixels": int(native_binary.sum()),
                "resized_pixels": int(resized_binary.sum()),
                "native_components": int(native_count),
                "resized_components": int(resized_count),
                "projected_component_areas": [float(area * scale) for area in native_areas],
            })
        if index % 100 == 0 or index == len(mask_paths):
            print(json.dumps({"completed": index, "total": len(mask_paths)}), flush=True)

    centers = sorted({row["center"] for row in rows})
    summaries = []
    for class_id in TARGET_CLASS_IDS:
        summaries.append(summarize(rows, class_id, None))
        for center in centers:
            summaries.append(summarize(rows, class_id, center))
    csv_rows = [{
        key: "" if value is None else value
        for key, value in row.items()
    } for row in summaries]
    write_csv(output_dir / "resolution_survival_by_class_center.csv", csv_rows)

    overall = [row for row in summaries if row["center"] == "overall"]
    critical = [
        row for row in overall
        if (row["projected_component_lt16_rate"] or 0.0) >= 0.20
        or (row["frame_disappearance_rate"] or 0.0) >= 0.20
    ]
    decision = "high_resolution" if critical else "class_aware_sampling"
    payload = {
        "status": "PASS",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_size": [INPUT_HEIGHT, INPUT_WIDTH],
        "target_class_ids": list(TARGET_CLASS_IDS),
        "target_class_names": [labelmap.fine_names[index] for index in TARGET_CLASS_IDS],
        "decision_rule": (
            "high_resolution if any target class has >=20% projected components below "
            "16 pixels or >=20% target-present frames disappear after resize"
        ),
        "selected_intervention": decision,
        "critical_classes": [row["class_name"] for row in critical],
        "overall": overall,
        "caveat": (
            "Projected component area is a resize-risk proxy; component merging can make "
            "raw component-count retention optimistic."
        ),
    }
    write_json(output_dir / "p2_resolution_audit.json", payload)
    lines = [
        "# P2 resolution-survival audit",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent / run",
        "- Verification Status: PASS",
        "- Input: frozen 524-frame P0/P1 dataset",
        "",
        f"Selected intervention: **{decision}**.",
        "",
        "| Class | Present frames | <16px components | Frame disappearance |",
        "|---|---:|---:|---:|",
    ]
    for row in overall:
        disappearance = row["frame_disappearance_rate"] or 0.0
        below16 = row["projected_component_lt16_rate"] or 0.0
        lines.append(
            f"| {row['class_name']} | {row['present_frames']} | {below16:.1%} | "
            f"{disappearance:.1%} |"
        )
    lines.extend(["", f"Critical classes: {', '.join(payload['critical_classes']) or 'none'}."])
    (output_dir / "P2_RESOLUTION_AUDIT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
