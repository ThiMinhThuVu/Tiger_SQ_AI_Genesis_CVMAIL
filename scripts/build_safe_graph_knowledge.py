#!/usr/bin/env python3
"""Build leakage-safe SAFE-Graph knowledge artifacts from TIGER ground truth.

The script deliberately separates:

* protocol-derived station/anatomy edges;
* empirical co-visibility in GT frames;
* geometry between simultaneously visible anatomy masks.

An empirical zero is never emitted as a forbidden anatomical relation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))
from tiger_models.data import (  # noqa: E402
    STATIONS,
    LabelMap,
    Record,
    case_from_name,
    load_visibility,
    records_for_fold,
    sha256,
)


ANATOMY_RELATION_EXCLUDE = {0, 1, 2, 24, 25}
KAPPA_CASES = 3.0


@dataclass
class BinarySupport:
    exposure_frames: int = 0
    positive_frames: int = 0
    by_case: dict[str, list[int]] = field(default_factory=lambda: defaultdict(lambda: [0, 0]))

    def observe(self, case_id: str, positive: bool) -> None:
        self.exposure_frames += 1
        self.positive_frames += int(positive)
        self.by_case[case_id][0] += 1
        self.by_case[case_id][1] += int(positive)

    def summary(self) -> dict[str, float | int]:
        exposure_cases = len(self.by_case)
        positive_cases = sum(item[1] > 0 for item in self.by_case.values())
        case_rates = [positive / exposure for exposure, positive in self.by_case.values()]
        frame_rate = self.positive_frames / max(self.exposure_frames, 1)
        return {
            "exposure_frames": self.exposure_frames,
            "positive_frames": self.positive_frames,
            "frame_probability": frame_rate,
            "beta11_frame_probability": (self.positive_frames + 1) / (self.exposure_frames + 2),
            "exposure_cases": exposure_cases,
            "positive_cases": positive_cases,
            "mean_within_case_probability": float(np.mean(case_rates)) if case_rates else 0.0,
            "case_support_weight": exposure_cases / (exposure_cases + KAPPA_CASES),
            "empirical_status": empirical_status(exposure_cases, positive_cases, frame_rate),
        }


@dataclass
class PairSupport(BinarySupport):
    distances: list[float] = field(default_factory=list)
    dx_values: list[float] = field(default_factory=list)
    dy_values: list[float] = field(default_factory=list)
    log_area_ratios: list[float] = field(default_factory=list)
    touches: list[int] = field(default_factory=list)

    def observe_pair(
        self,
        case_id: str,
        present: bool,
        distance: float | None = None,
        dx: float | None = None,
        dy: float | None = None,
        log_area_ratio: float | None = None,
        touch: bool | None = None,
    ) -> None:
        self.observe(case_id, present)
        if not present:
            return
        assert None not in (distance, dx, dy, log_area_ratio, touch)
        self.distances.append(float(distance))
        self.dx_values.append(float(dx))
        self.dy_values.append(float(dy))
        self.log_area_ratios.append(float(log_area_ratio))
        self.touches.append(int(bool(touch)))

    def relation_summary(self) -> dict[str, float | int | None]:
        base = self.summary()
        base.update({
            "relation_observations": len(self.distances),
            "boundary_distance_norm_mean": mean_or_none(self.distances),
            "boundary_distance_norm_q05": quantile_or_none(self.distances, 0.05),
            "boundary_distance_norm_q50": quantile_or_none(self.distances, 0.50),
            "boundary_distance_norm_q95": quantile_or_none(self.distances, 0.95),
            "centroid_dx_norm_mean": mean_or_none(self.dx_values),
            "centroid_dx_norm_sd": sample_sd_or_none(self.dx_values),
            "centroid_dy_norm_mean": mean_or_none(self.dy_values),
            "centroid_dy_norm_sd": sample_sd_or_none(self.dy_values),
            "log_area_ratio_mean": mean_or_none(self.log_area_ratios),
            "touch_probability_when_covisible": mean_or_none(self.touches),
            "view_relation_eligible": view_relation_eligible(self.dx_values, self.dy_values, self.by_case),
        })
        return base


def empirical_status(exposure_cases: int, positive_cases: int, frame_rate: float) -> str:
    if exposure_cases < 3:
        return "insufficient_case_support"
    if positive_cases == 0:
        return "unobserved_not_forbidden"
    if exposure_cases >= 5 and positive_cases >= 3 and frame_rate >= 0.20:
        return "supported_soft_association"
    return "weak_soft_association"


def mean_or_none(values: Iterable[float]) -> float | None:
    values = list(values)
    return float(np.mean(values)) if values else None


def sample_sd_or_none(values: Iterable[float]) -> float | None:
    values = list(values)
    return float(statistics.stdev(values)) if len(values) >= 2 else None


def quantile_or_none(values: Iterable[float], q: float) -> float | None:
    values = list(values)
    return float(np.quantile(values, q)) if values else None


def view_relation_eligible(
    dx_values: list[float], dy_values: list[float], by_case: dict[str, list[int]]
) -> bool:
    """Conservative prefilter; clinical review is still required."""
    if len(by_case) < 5 or len(dx_values) < 10:
        return False
    dx = np.asarray(dx_values)
    dy = np.asarray(dy_values)
    dominant_dx = max(float((dx > 0).mean()), float((dx < 0).mean()))
    dominant_dy = max(float((dy > 0).mean()), float((dy < 0).mean()))
    return max(dominant_dx, dominant_dy) >= 0.80


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--seed",
        type=Path,
        default=ROOT / "knowledge_graph" / "clinical_station_anatomy_seed_v1.json",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--all-folds", action="store_true")
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--include-all-data-descriptive", action="store_true")
    parser.add_argument("--relation-max-side", type=int, default=512)
    parser.add_argument("--touch-pixels", type=float, default=2.0)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def all_records(data_root: Path) -> list[Record]:
    visibility = load_visibility(data_root / "lymph_node_station_visibility.csv")
    records: list[Record] = []
    for image in sorted((data_root / "images").glob("*.png")):
        if image.name not in visibility:
            raise FileNotFoundError(f"Missing visibility annotation for {image.name}")
        mask = data_root / "masks_fine" / image.name
        if not mask.is_file():
            raise FileNotFoundError(mask)
        records.append(Record(image.name, case_from_name(image.name), image, mask, visibility[image.name]))
    if len(records) != 140 or len({record.case_id for record in records}) != 10:
        raise ValueError("Expected exactly 140 frames from 10 cases")
    return records


def annotated_station(record: Record) -> str:
    station = Path(record.name).stem.removeprefix(record.case_id + "_")
    if station not in STATIONS:
        raise ValueError(f"Cannot parse annotated station from {record.name}")
    return station


def station_contexts(record: Record) -> list[tuple[str, str]]:
    contexts = [("annotated_station", annotated_station(record))]
    contexts.extend(
        ("visible_station", station)
        for station, visible in zip(STATIONS, record.visibility)
        if bool(visible)
    )
    return contexts


def resize_ids(ids: np.ndarray, max_side: int) -> np.ndarray:
    height, width = ids.shape
    if max(height, width) <= max_side:
        return ids
    scale = max_side / max(height, width)
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    return np.asarray(Image.fromarray(ids.astype(np.uint8)).resize(target, Image.Resampling.NEAREST))


def mask_geometry(ids: np.ndarray, class_ids: list[int], touch_pixels: float) -> dict[tuple[int, int], dict]:
    height, width = ids.shape
    diagonal = math.hypot(height, width)
    masks = {class_id: ids == class_id for class_id in class_ids}
    centroids: dict[int, tuple[float, float]] = {}
    areas: dict[int, int] = {}
    distance_maps: dict[int, np.ndarray] = {}
    for class_id, mask in masks.items():
        ys, xs = np.nonzero(mask)
        centroids[class_id] = (float(xs.mean()), float(ys.mean()))
        areas[class_id] = int(mask.sum())
        distance_maps[class_id] = ndimage.distance_transform_edt(~mask)
    output: dict[tuple[int, int], dict] = {}
    for first, second in combinations(class_ids, 2):
        distance = min(
            float(distance_maps[first][masks[second]].min()),
            float(distance_maps[second][masks[first]].min()),
        )
        first_x, first_y = centroids[first]
        second_x, second_y = centroids[second]
        output[(first, second)] = {
            "boundary_distance_norm": distance / max(diagonal, 1.0),
            "centroid_dx_norm": (second_x - first_x) / max(width, 1),
            "centroid_dy_norm": (second_y - first_y) / max(height, 1),
            "log_area_ratio": math.log((areas[second] + 1) / (areas[first] + 1)),
            "touch": distance <= touch_pixels,
        }
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_fine_types(labelmap_path: Path) -> dict[int, str]:
    with labelmap_path.open(newline="") as handle:
        return {int(row["fine_id"]): row["type"] for row in csv.DictReader(handle)}


def build_scope(
    records: list[Record],
    labelmap: LabelMap,
    fine_types: dict[int, str],
    seed: dict,
    output_dir: Path,
    scope_id: str,
    model_use_allowed: bool,
    relation_max_side: int,
    touch_pixels: float,
) -> dict:
    station_anatomy: dict[tuple[str, str, int], BinarySupport] = defaultdict(BinarySupport)
    station_pairs: dict[tuple[str, str, int, int], PairSupport] = defaultdict(PairSupport)
    station_exposure: dict[tuple[str, str], BinarySupport] = defaultdict(BinarySupport)
    station_covis: dict[tuple[str, str], BinarySupport] = defaultdict(BinarySupport)
    frame_rows: list[dict] = []

    relation_class_ids = [
        class_id for class_id in range(31)
        if fine_types[class_id] == "Anatomical" and class_id not in ANATOMY_RELATION_EXCLUDE
    ]

    decoded: list[tuple[Record, np.ndarray, set[int], dict]] = []
    for record in records:
        rgb = np.asarray(Image.open(record.fine_mask).convert("RGB"))
        ids = labelmap.decode_fine(rgb)
        present = {int(class_id) for class_id in np.unique(ids)}
        relation_present = sorted(present.intersection(relation_class_ids))
        small_ids = resize_ids(ids, relation_max_side)
        geometry = mask_geometry(small_ids, relation_present, touch_pixels)
        decoded.append((record, ids, present, geometry))

        visible_stations = [
            station for station, visible in zip(STATIONS, record.visibility) if bool(visible)
        ]
        areas = np.bincount(ids.ravel(), minlength=31)
        frame_rows.append({
            "scope_id": scope_id,
            "case_id": record.case_id,
            "frame": record.name,
            "annotated_station": annotated_station(record),
            "visible_stations": "|".join(visible_stations),
            "visible_station_count": len(visible_stations),
            "present_anatomy_ids": "|".join(map(str, sorted(present - {0, 1, 2}))),
            "present_anatomy_names": "|".join(labelmap.fine_names[i] for i in sorted(present - {0, 1, 2})),
            "instrument_fraction": float(areas[1] / ids.size),
            "other_fraction": float(areas[2] / ids.size),
            "image_height": ids.shape[0],
            "image_width": ids.shape[1],
        })

    for record, _ids, present, geometry in decoded:
        contexts = station_contexts(record)
        for basis, station in contexts:
            station_exposure[(basis, station)].observe(record.case_id, True)
            for class_id in range(1, 31):
                station_anatomy[(basis, station, class_id)].observe(record.case_id, class_id in present)
            for first, second in combinations(relation_class_ids, 2):
                pair_present = first in present and second in present
                metrics = geometry.get((first, second), {})
                station_pairs[(basis, station, first, second)].observe_pair(
                    record.case_id,
                    pair_present,
                    metrics.get("boundary_distance_norm"),
                    metrics.get("centroid_dx_norm"),
                    metrics.get("centroid_dy_norm"),
                    metrics.get("log_area_ratio"),
                    metrics.get("touch"),
                )

        visible = {station for station, flag in zip(STATIONS, record.visibility) if bool(flag)}
        for first, second in combinations(STATIONS, 2):
            station_covis[(first, second)].observe(record.case_id, first in visible and second in visible)

    station_anatomy_rows: list[dict] = []
    support_lookup: dict[tuple[str, str], dict] = {}
    for (basis, station, class_id), support in sorted(station_anatomy.items()):
        summary = support.summary()
        row = {
            "scope_id": scope_id,
            "context_basis": basis,
            "station": station,
            "anatomy_id": class_id,
            "anatomy_name": labelmap.fine_names[class_id],
            "relation": "co_visible_with",
            **summary,
            "zero_semantics": "unobserved_not_forbidden" if support.positive_frames == 0 else "observed",
        }
        station_anatomy_rows.append(row)
        if basis == "visible_station":
            support_lookup[(station, labelmap.fine_names[class_id])] = summary

    pair_rows: list[dict] = []
    for (basis, station, first, second), support in sorted(station_pairs.items()):
        pair_rows.append({
            "scope_id": scope_id,
            "context_basis": basis,
            "station": station,
            "anatomy_i_id": first,
            "anatomy_i_name": labelmap.fine_names[first],
            "anatomy_j_id": second,
            "anatomy_j_name": labelmap.fine_names[second],
            "relation": "station_conditioned_pair_geometry",
            **support.relation_summary(),
        })

    covis_rows: list[dict] = []
    for (first, second), support in sorted(station_covis.items()):
        covis_rows.append({
            "scope_id": scope_id,
            "station_i": first,
            "station_j": second,
            "relation": "co_visible_in_frame",
            **support.summary(),
        })

    alignment_rows: list[dict] = []
    defaults = seed["edge_defaults"]
    for index, edge in enumerate(seed["station_anatomy_edges"], start=1):
        support = support_lookup.get((edge["station"], edge["anatomy"]), {})
        alignment_rows.append({
            "rule_id": f"SA-{index:03d}",
            "station": edge["station"],
            "anatomy": edge["anatomy"],
            "protocol_relation": edge["relation"],
            "review_status": edge.get("review_status", defaults["review_status"]),
            "constraint_type": edge.get("constraint_type", defaults["constraint_type"]),
            "warning_eligible": edge.get("warning_eligible", defaults["warning_eligible"]),
            "correction_eligible": edge.get("correction_eligible", defaults["correction_eligible"]),
            "sources": "|".join(edge["sources"]),
            "gt_exposure_cases": support.get("exposure_cases", 0),
            "gt_positive_cases": support.get("positive_cases", 0),
            "gt_frame_probability": support.get("frame_probability", 0.0),
            "gt_empirical_status": support.get("empirical_status", "not_available"),
            "interpretation": "GT supports/does not support visibility only; it cannot invalidate protocol anatomy.",
        })

    write_csv(output_dir / "frame_observations.csv", frame_rows)
    write_csv(output_dir / "station_anatomy_support.csv", station_anatomy_rows)
    write_csv(output_dir / "anatomy_pair_relations.csv", pair_rows)
    write_csv(output_dir / "station_co_visibility.csv", covis_rows)
    write_csv(output_dir / "expert_empirical_alignment.csv", alignment_rows)

    manifest = {
        "scope_id": scope_id,
        "model_use_allowed": model_use_allowed,
        "model_use_note": (
            "May be used only by the corresponding outer fold."
            if model_use_allowed else
            "Descriptive expert-audit artifact only; must not be used for cross-validated model fitting."
        ),
        "frames": len(records),
        "cases": len({record.case_id for record in records}),
        "case_ids": sorted({record.case_id for record in records}),
        "relation_max_side": relation_max_side,
        "touch_pixels_at_relation_resolution": touch_pixels,
        "semantic_guardrails": [
            "co_visible_with is not contains",
            "empirical zero is not forbidden",
            "absence is not a violation",
            "view-dependent directions are soft and require stability plus clinical review",
            "no hidden anatomy is inferred beneath an instrument from a single frame",
        ],
        "outputs": [
            "frame_observations.csv",
            "station_anatomy_support.csv",
            "anatomy_pair_relations.csv",
            "station_co_visibility.csv",
            "expert_empirical_alignment.csv",
        ],
    }
    (output_dir / "scope_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def validate_seed(seed: dict, labelmap: LabelMap) -> None:
    if seed.get("schema_version") != "1.0.0":
        raise ValueError("Unsupported seed schema")
    stations = [item["station_id"] for item in seed["stations"]]
    if stations != list(STATIONS):
        raise ValueError(f"Seed stations do not match runtime station order: {stations}")
    valid_names = set(labelmap.fine_names)
    seen: set[tuple[str, str, str]] = set()
    for edge in seed["station_anatomy_edges"]:
        key = (edge["station"], edge["anatomy"], edge["relation"])
        if key in seen:
            raise ValueError(f"Duplicate seed edge: {key}")
        seen.add(key)
        if edge["station"] not in STATIONS:
            raise ValueError(f"Unknown station in seed: {edge}")
        if edge["anatomy"] not in valid_names:
            raise ValueError(f"Anatomy does not match labelmap: {edge}")
        if edge["relation"] != "boundary_structure_of":
            raise ValueError(f"Unexpected seed relation: {edge}")
    required = {
        ("13L", "Left inferior pulmonary ligament"),
        ("13R", "Right inferior pulmonary ligament"),
    }
    actual = {(edge["station"], edge["anatomy"]) for edge in seed["station_anatomy_edges"]}
    if not required <= actual:
        raise ValueError("Corrected pulmonary-ligament mapping is missing")
    forbidden_reversed = {
        ("13R", "Left inferior pulmonary ligament"),
        ("13L", "Right inferior pulmonary ligament"),
    }
    if actual & forbidden_reversed:
        raise ValueError("Reversed pulmonary-ligament mapping must not be active")


def main() -> None:
    args = parse_args()
    if args.fold is not None and args.all_folds:
        raise ValueError("Choose --fold or --all-folds, not both")
    if args.fold is None and not args.all_folds:
        raise ValueError("One of --fold or --all-folds is required")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output root: {args.output_root}")

    labelmap = LabelMap.load(args.data_root / "labelmap.csv")
    fine_types = load_fine_types(args.data_root / "labelmap.csv")
    seed = json.loads(args.seed.read_text())
    validate_seed(seed, labelmap)
    args.output_root.mkdir(parents=True, exist_ok=True)

    manifests: list[dict] = []
    folds = range(5) if args.all_folds else [args.fold]
    for fold in folds:
        records = records_for_fold(args.data_root, int(fold), "train")
        manifests.append(build_scope(
            records,
            labelmap,
            fine_types,
            seed,
            args.output_root / f"outer_fold_{fold}_train",
            f"outer_fold_{fold}_train",
            True,
            args.relation_max_side,
            args.touch_pixels,
        ))

    if args.include_all_data_descriptive:
        manifests.append(build_scope(
            all_records(args.data_root),
            labelmap,
            fine_types,
            seed,
            args.output_root / "all_data_descriptive_DO_NOT_FIT",
            "all_data_descriptive_DO_NOT_FIT",
            False,
            args.relation_max_side,
            args.touch_pixels,
        ))

    root_manifest = {
        "artifact": "SAFE-Graph hybrid knowledge evidence",
        "artifact_version": "kg_v1_20260808",
        "created_by": "scripts/build_safe_graph_knowledge.py",
        "seed_path": str(args.seed),
        "seed_sha256": file_sha256(args.seed),
        "labelmap_sha256": sha256(args.data_root / "labelmap.csv"),
        "visibility_sha256": sha256(args.data_root / "lymph_node_station_visibility.csv"),
        "scopes": manifests,
        "promotion_status": "NOT_RUNTIME_APPROVED",
        "promotion_note": "Protocol edges require clinical review and verifier thresholds require validation.",
    }
    (args.output_root / "manifest.json").write_text(json.dumps(root_manifest, indent=2) + "\n")
    print(json.dumps({
        "output_root": str(args.output_root),
        "scopes": [item["scope_id"] for item in manifests],
        "status": "completed",
    }, indent=2))


if __name__ == "__main__":
    main()
