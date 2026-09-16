#!/usr/bin/env python3
"""Prepare auditable expert-review sheets from SAFE-Graph KG artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REVIEW_FIELDS = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kg-root", type=Path,
        default=ROOT / "artifacts/safe_graph_knowledge/kg_v1_20260808",
    )
    parser.add_argument(
        "--station-seed", type=Path,
        default=ROOT / "knowledge_graph/clinical_station_anatomy_seed_v1.json",
    )
    parser.add_argument(
        "--anatomy-seed", type=Path,
        default=ROOT / "knowledge_graph/clinical_anatomy_relations_seed_v1.json",
    )
    parser.add_argument("--labelmap", type=Path, default=ROOT / "data/labelmap.csv")
    parser.add_argument("--oof-confusion", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty review sheet: {path}")
    fields = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def with_blank_review_fields(row: dict) -> dict:
    return {**row, **{field: "" for field in REVIEW_FIELDS}}


def load_labelmap(path: Path) -> tuple[dict[str, dict], dict[int, dict]]:
    rows = read_csv(path)
    by_name = {row["fine_name"]: row for row in rows}
    by_id = {int(row["fine_id"]): row for row in rows}
    return by_name, by_id


def priority(clinical_weight: float, empirical_status: str, probability: float) -> str:
    if clinical_weight >= 3 and (empirical_status != "supported_soft_association" or probability < 0.20):
        return "P0"
    if clinical_weight >= 3 or empirical_status in {"unobserved_not_forbidden", "weak_soft_association"}:
        return "P1"
    return "P2"


def station_review_rows(
    kg_root: Path, station_seed: dict, label_by_name: dict[str, dict], label_by_id: dict[int, dict]
) -> list[dict]:
    audit_dir = kg_root / "all_data_descriptive_DO_NOT_FIT"
    alignment = read_csv(audit_dir / "expert_empirical_alignment.csv")
    support = read_csv(audit_dir / "station_anatomy_support.csv")
    protocol_pairs = {(row["station"], row["anatomy"]) for row in alignment}
    station_independent = set(station_seed["station_independent_context_classes"])
    output: list[dict] = []

    for row in alignment:
        label = label_by_name[row["anatomy"]]
        weight = float(label["weight"])
        probability = float(row["gt_frame_probability"])
        output.append(with_blank_review_fields({
            "candidate_type": "protocol_edge",
            "review_priority": priority(weight, row["gt_empirical_status"], probability),
            "rule_id": row["rule_id"],
            "station": row["station"],
            "anatomy_id": label["fine_id"],
            "anatomy": row["anatomy"],
            "clinical_weight": label["weight"],
            "protocol_relation": row["protocol_relation"],
            "protocol_review_status": row["review_status"],
            "sources": row["sources"],
            "gt_exposure_cases": row["gt_exposure_cases"],
            "gt_positive_cases": row["gt_positive_cases"],
            "gt_frame_probability": row["gt_frame_probability"],
            "gt_empirical_status": row["gt_empirical_status"],
            "automatic_semantics": "GT association cannot promote a hard rule",
        }))

    empirical_candidates: list[dict] = []
    for row in support:
        if row["context_basis"] != "visible_station":
            continue
        class_id = int(row["anatomy_id"])
        label = label_by_id[class_id]
        pair = (row["station"], row["anatomy_name"])
        if (
            pair in protocol_pairs
            or label["type"] != "Anatomical"
            or class_id in {24, 25}
            or row["anatomy_name"] in station_independent
        ):
            continue
        exposure_cases = int(row["exposure_cases"])
        positive_cases = int(row["positive_cases"])
        probability = float(row["frame_probability"])
        if exposure_cases < 5 or positive_cases < 3 or probability < 0.20:
            continue
        weight = float(label["weight"])
        empirical_candidates.append(with_blank_review_fields({
            "candidate_type": "empirical_only_descriptive",
            "review_priority": priority(weight, row["empirical_status"], probability),
            "rule_id": "",
            "station": row["station"],
            "anatomy_id": class_id,
            "anatomy": row["anatomy_name"],
            "clinical_weight": label["weight"],
            "protocol_relation": "",
            "protocol_review_status": "not_in_protocol_seed",
            "sources": "all_data_descriptive_DO_NOT_FIT",
            "gt_exposure_cases": row["exposure_cases"],
            "gt_positive_cases": row["positive_cases"],
            "gt_frame_probability": row["frame_probability"],
            "gt_empirical_status": row["empirical_status"],
            "automatic_semantics": "Empirical co-visibility candidate; not anatomy containment",
        }))
    empirical_candidates.sort(key=lambda row: (
        row["review_priority"], -float(row["clinical_weight"]), -float(row["gt_frame_probability"]),
        row["station"], row["anatomy"],
    ))
    # Limit exploratory correlations to five per station so clinical review stays
    # actionable. Complete empirical tables remain in the immutable KG artifact.
    station_counts: dict[str, int] = {}
    selected_empirical: list[dict] = []
    for row in empirical_candidates:
        count = station_counts.get(row["station"], 0)
        if count >= 5:
            continue
        station_counts[row["station"]] = count + 1
        selected_empirical.append(row)
    return sorted(output, key=lambda row: (row["review_priority"], row["station"], row["anatomy"])) + selected_empirical


def empirical_signal(row: dict[str, str]) -> str:
    touch = row["touch_probability_when_covisible"]
    q95 = row["boundary_distance_norm_q95"]
    view = row["view_relation_eligible"].lower() == "true"
    if touch and float(touch) >= 0.75:
        return "frequent_touch_candidate"
    if q95 and float(q95) <= 0.03:
        return "near_candidate"
    if view:
        return "stable_image_order_candidate"
    return "co_visibility_only"


def anatomy_review_rows(
    kg_root: Path, anatomy_seed: dict, label_by_name: dict[str, dict], label_by_id: dict[int, dict]
) -> list[dict]:
    relations = read_csv(kg_root / "all_data_descriptive_DO_NOT_FIT/anatomy_pair_relations.csv")
    visible_rows = [row for row in relations if row["context_basis"] == "visible_station"]
    lookup: dict[tuple[str, frozenset[str]], dict] = {
        (row["station"], frozenset((row["anatomy_i_name"], row["anatomy_j_name"]))): row
        for row in visible_rows
    }
    seed_keys: set[tuple[str, frozenset[str]]] = set()
    output: list[dict] = []
    defaults = anatomy_seed["defaults"]
    for rule in anatomy_seed["relations"]:
        pair = frozenset((rule["subject"], rule["object"]))
        stations = rule["stations"] or ["ALL"]
        for station in stations:
            seed_keys.add((station, pair))
            empirical = lookup.get((station, pair), {}) if station != "ALL" else {}
            first = label_by_name[rule["subject"]]
            second = label_by_name[rule["object"]]
            max_weight = max(float(first["weight"]), float(second["weight"]))
            status = empirical.get("empirical_status", "not_observed_in_applicable_station")
            probability = float(empirical.get("frame_probability", 0.0) or 0.0)
            output.append(with_blank_review_fields({
                "candidate_type": "protocol_relation",
                "review_priority": priority(max_weight, status, probability),
                "rule_id": rule["rule_id"],
                "station": station,
                "subject_id": first["fine_id"],
                "subject": rule["subject"],
                "protocol_relation": rule["relation"],
                "object_id": second["fine_id"],
                "object": rule["object"],
                "max_clinical_weight": max_weight,
                "clinical_scope": rule["clinical_scope"],
                "single_frame_observability": rule["single_frame_observability"],
                "operational_candidate": rule["operational_candidate"],
                "sources": "|".join(rule["sources"]),
                "seed_review_status": rule.get("review_status", defaults["review_status"]),
                "gt_exposure_cases": empirical.get("exposure_cases", ""),
                "gt_positive_cases": empirical.get("positive_cases", ""),
                "gt_copresence_probability": empirical.get("frame_probability", ""),
                "gt_distance_q05": empirical.get("boundary_distance_norm_q05", ""),
                "gt_distance_q50": empirical.get("boundary_distance_norm_q50", ""),
                "gt_distance_q95": empirical.get("boundary_distance_norm_q95", ""),
                "gt_touch_probability": empirical.get("touch_probability_when_covisible", ""),
                "gt_dx_mean": empirical.get("centroid_dx_norm_mean", ""),
                "gt_dx_sd": empirical.get("centroid_dx_norm_sd", ""),
                "gt_dy_mean": empirical.get("centroid_dy_norm_mean", ""),
                "gt_dy_sd": empirical.get("centroid_dy_norm_sd", ""),
                "gt_view_relation_eligible": empirical.get("view_relation_eligible", ""),
                "automatic_empirical_signal": empirical_signal(empirical) if empirical else "none",
                "automatic_semantics": "Candidate only; no automatic promotion",
            }))

    empirical_candidates: list[dict] = []
    for row in visible_rows:
        first_id, second_id = int(row["anatomy_i_id"]), int(row["anatomy_j_id"])
        first, second = label_by_id[first_id], label_by_id[second_id]
        pair = frozenset((row["anatomy_i_name"], row["anatomy_j_name"]))
        if (row["station"], pair) in seed_keys:
            continue
        max_weight = max(float(first["weight"]), float(second["weight"]))
        signal = empirical_signal(row)
        if (
            int(row["exposure_cases"]) < 5
            or int(row["positive_cases"]) < 5
            or int(row["positive_frames"]) < 10
            or float(row["frame_probability"]) < 0.20
            or max_weight < 3
            or signal not in {"frequent_touch_candidate", "near_candidate"}
        ):
            continue
        empirical_candidates.append(with_blank_review_fields({
            "candidate_type": "empirical_only_descriptive",
            "review_priority": priority(max_weight, row["empirical_status"], float(row["frame_probability"])),
            "rule_id": "",
            "station": row["station"],
            "subject_id": first_id,
            "subject": row["anatomy_i_name"],
            "protocol_relation": "",
            "object_id": second_id,
            "object": row["anatomy_j_name"],
            "max_clinical_weight": max_weight,
            "clinical_scope": "unknown_empirical_candidate",
            "single_frame_observability": "yes_for_observed_geometry",
            "operational_candidate": signal,
            "sources": "all_data_descriptive_DO_NOT_FIT",
            "seed_review_status": "not_in_protocol_seed",
            "gt_exposure_cases": row["exposure_cases"],
            "gt_positive_cases": row["positive_cases"],
            "gt_copresence_probability": row["frame_probability"],
            "gt_distance_q05": row["boundary_distance_norm_q05"],
            "gt_distance_q50": row["boundary_distance_norm_q50"],
            "gt_distance_q95": row["boundary_distance_norm_q95"],
            "gt_touch_probability": row["touch_probability_when_covisible"],
            "gt_dx_mean": row["centroid_dx_norm_mean"],
            "gt_dx_sd": row["centroid_dx_norm_sd"],
            "gt_dy_mean": row["centroid_dy_norm_mean"],
            "gt_dy_sd": row["centroid_dy_norm_sd"],
            "gt_view_relation_eligible": row["view_relation_eligible"],
            "automatic_empirical_signal": signal,
            "automatic_semantics": "Empirical candidate; may reflect station confounding",
        }))
    empirical_candidates.sort(key=lambda row: (
        row["review_priority"], -float(row["max_clinical_weight"]),
        -float(row["gt_copresence_probability"]), row["station"], row["subject"], row["object"],
    ))
    return sorted(output, key=lambda row: (row["review_priority"], row["rule_id"], row["station"])) + empirical_candidates


def confusion_review_rows(path: Path) -> tuple[list[dict], dict]:
    graph = json.loads(path.read_text())
    if graph.get("split") != "out_of_fold_validation_only" or graph.get("test_data_used") is not False:
        raise ValueError("Confusion graph must be validation-only and test-free")
    source_files = [Path(item) for item in graph.get("source_files", [])]
    if len(source_files) != 5 or not all(path.is_file() for path in source_files):
        raise ValueError("Confusion graph must cite five existing fold validation files")
    rows: list[dict] = []
    for index, edge in enumerate(graph["edges"], start=1):
        risk = float(edge["clinical_risk"])
        weight = float(edge["edge_weight"])
        rows.append(with_blank_review_fields({
            "candidate_type": "oof_confusion_pair",
            "review_priority": "P0" if risk >= 3 and weight >= 0.10 else ("P1" if risk >= 2 else "P2"),
            "confusion_rule_id": f"CF-{index:03d}",
            "class_i": edge["class_i"],
            "name_i": edge["name_i"],
            "class_j": edge["class_j"],
            "name_j": edge["name_j"],
            "edge_weight": edge["edge_weight"],
            "i_to_j_rate": edge["i_to_j_rate"],
            "j_to_i_rate": edge["j_to_i_rate"],
            "i_to_j_pixels": edge["i_to_j_pixels"],
            "j_to_i_pixels": edge["j_to_i_pixels"],
            "clinical_risk": edge["clinical_risk"],
            "source_model_dir": graph["source_model_dir"],
            "source_split": graph["split"],
            "test_data_used": graph["test_data_used"],
            "automatic_semantics": "Pixel confusion candidate; component-level confirmation still required",
        }))
    return rows, graph


def validate_seed_names(seed: dict, label_by_name: dict[str, dict]) -> None:
    valid = set(label_by_name)
    for rule in seed["relations"]:
        if rule["subject"] not in valid or rule["object"] not in valid:
            raise ValueError(f"Relation seed name does not match labelmap: {rule}")
        if rule["warning_eligible"] if "warning_eligible" in rule else False:
            raise ValueError("Seed v1 cannot contain active warning edges")


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty review directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    station_seed = json.loads(args.station_seed.read_text())
    anatomy_seed = json.loads(args.anatomy_seed.read_text())
    label_by_name, label_by_id = load_labelmap(args.labelmap)
    validate_seed_names(anatomy_seed, label_by_name)

    station_rows = station_review_rows(args.kg_root, station_seed, label_by_name, label_by_id)
    anatomy_rows = anatomy_review_rows(args.kg_root, anatomy_seed, label_by_name, label_by_id)
    write_csv(args.output_dir / "station_anatomy_review.csv", station_rows)
    write_csv(args.output_dir / "anatomy_relation_review.csv", anatomy_rows)

    confusion_rows: list[dict] = []
    confusion_graph: dict = {}
    if args.oof_confusion is not None:
        confusion_rows, confusion_graph = confusion_review_rows(args.oof_confusion)
        write_csv(args.output_dir / "confusion_risk_review.csv", confusion_rows)

    manifest = {
        "artifact": "SAFE-Graph expert review package",
        "version": "expert_review_v1",
        "created_utc": "2026-08-08",
        "input_hashes": {
            "station_seed": sha256(args.station_seed),
            "anatomy_seed": sha256(args.anatomy_seed),
            "labelmap": sha256(args.labelmap),
            "oof_confusion": sha256(args.oof_confusion) if args.oof_confusion else None,
        },
        "counts": {
            "station_review_rows": len(station_rows),
            "station_protocol_rows": sum(row["candidate_type"] == "protocol_edge" for row in station_rows),
            "station_empirical_only_rows": sum(row["candidate_type"] == "empirical_only_descriptive" for row in station_rows),
            "anatomy_review_rows": len(anatomy_rows),
            "anatomy_protocol_rows": sum(row["candidate_type"] == "protocol_relation" for row in anatomy_rows),
            "anatomy_empirical_only_rows": sum(row["candidate_type"] == "empirical_only_descriptive" for row in anatomy_rows),
            "confusion_review_rows": len(confusion_rows),
        },
        "oof_source": {
            "source_model_dir": confusion_graph.get("source_model_dir"),
            "split": confusion_graph.get("split"),
            "test_data_used": confusion_graph.get("test_data_used"),
            "source_fold_files": confusion_graph.get("source_files", []),
        },
        "status": "PENDING_EXPERT_REVIEW",
        "guardrails": [
            "Blank expert fields must be completed by a named reviewer",
            "All-data empirical candidates are descriptive and cannot fit the verifier",
            "No candidate is automatically warning- or correction-eligible",
            "Pixel confusion pairs require component-level confirmation",
        ],
    }
    (args.output_dir / "review_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output_dir": str(args.output_dir), **manifest["counts"]}, indent=2))


if __name__ == "__main__":
    main()
