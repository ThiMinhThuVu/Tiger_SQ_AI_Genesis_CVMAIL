#!/usr/bin/env python3
"""Validate completed SAFE-Graph clinical pilot review tables without promoting rules."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def validate_component_review(records: list[dict[str, str]]) -> list[str]:
    errors = []
    allowed = {
        "clinical_error_valid_yes_no_uncertain": {"YES", "NO", "UNCERTAIN"},
        "clinical_severity_0_1_2_3": {"0", "1", "2", "3"},
        "warning_would_be_useful_yes_no_uncertain": {"YES", "NO", "UNCERTAIN"},
        "tool_occluded_yes_no_uncertain": {"YES", "NO", "UNCERTAIN"},
        "camera_ambiguous_yes_no_uncertain": {"YES", "NO", "UNCERTAIN"},
    }
    for row in records:
        review_id = row["review_id"]
        for field, choices in allowed.items():
            value = row[field].strip().upper()
            if value and value not in choices:
                errors.append(f"{review_id}: invalid {field}={row[field]!r}")
        decision = row["clinical_error_valid_yes_no_uncertain"].strip()
        if decision:
            required = ["clinical_severity_0_1_2_3", "reviewer_id", "review_date"]
            for field in required:
                if not row[field].strip():
                    errors.append(f"{review_id}: reviewed row missing {field}")
            if row["reviewer_id"].strip().upper().startswith("AI_PROVISIONAL"):
                if not row["review_comment"].strip().startswith("AI_PROVISIONAL:"):
                    errors.append(f"{review_id}: AI review_comment must start with AI_PROVISIONAL:")
    return errors


def validate_kg_review(records: list[dict[str, str]], id_field: str) -> list[str]:
    errors = []
    allowed = {
        "expert_decision_valid_invalid_depends": {"VALID", "INVALID", "DEPENDS"},
        "expert_constraint_hard_soft": {"HARD", "SOFT"},
        "expert_absence_is_violation": {"YES", "NO", "DEPENDS"},
        "expert_warning_eligible": {"YES", "NO", "DEPENDS"},
        "expert_correction_eligible": {"YES", "NO", "DEPENDS"},
        "expert_camera_dependency": {"YES", "NO", "DEPENDS"},
        "expert_tool_occlusion_dependency": {"YES", "NO", "DEPENDS"},
    }
    for row in records:
        row_id = row[id_field]
        decision = row["expert_decision_valid_invalid_depends"].strip().upper()
        if decision and decision not in allowed["expert_decision_valid_invalid_depends"]:
            errors.append(f"{row_id}: invalid expert decision {decision!r}")
        for field, choices in allowed.items():
            value = row[field].strip().upper()
            if value and value not in choices:
                errors.append(f"{row_id}: invalid {field}={row[field]!r}")
        if decision:
            required = [
                "expert_relation_scope", "expert_constraint_hard_soft",
                "expert_warning_eligible", "expert_correction_eligible",
                "expert_camera_dependency", "expert_tool_occlusion_dependency",
                "expert_exception", "reviewer_id", "review_date",
            ]
            for field in required:
                if not row[field].strip():
                    errors.append(f"{row_id}: reviewed row missing {field}")
            if row["reviewer_id"].strip().upper().startswith("AI_PROVISIONAL"):
                if row["expert_constraint_hard_soft"].strip().upper() != "SOFT":
                    errors.append(f"{row_id}: AI provisional constraint must be SOFT")
                if row["expert_absence_is_violation"].strip().upper() != "NO":
                    errors.append(f"{row_id}: AI provisional absence_is_violation must be NO")
                if row["expert_correction_eligible"].strip().upper() != "NO":
                    errors.append(f"{row_id}: AI provisional correction_eligible must be NO")
                if not row["review_comment"].strip().startswith("AI_PROVISIONAL:"):
                    errors.append(f"{row_id}: AI review_comment must start with AI_PROVISIONAL:")
    return errors


def validate(review_root: Path) -> dict:
    component = rows(review_root / "component_error_review.csv")
    station = rows(review_root / "priority_station_review.csv")
    relation = rows(review_root / "priority_anatomy_relation_review.csv")
    confusion = rows(review_root / "priority_confusion_review.csv")
    errors = validate_component_review(component)
    errors += validate_kg_review(station, "rule_id")
    errors += validate_kg_review(relation, "rule_id")
    errors += validate_kg_review(confusion, "confusion_rule_id")
    reviewed = Counter()
    reviewed["component"] = sum(bool(row["clinical_error_valid_yes_no_uncertain"].strip()) for row in component)
    reviewed["station"] = sum(bool(row["expert_decision_valid_invalid_depends"].strip()) for row in station)
    reviewed["relation"] = sum(bool(row["expert_decision_valid_invalid_depends"].strip()) for row in relation)
    reviewed["confusion"] = sum(bool(row["expert_decision_valid_invalid_depends"].strip()) for row in confusion)
    totals = {
        "component": len(component), "station": len(station),
        "relation": len(relation), "confusion": len(confusion),
    }
    complete = not errors and all(reviewed[key] == totals[key] for key in totals)
    all_records = component + station + relation + confusion
    ai_provisional_rows = sum(
        row.get("reviewer_id", "").strip().upper().startswith("AI_PROVISIONAL")
        for row in all_records
    )
    if errors:
        status = "INVALID"
    elif ai_provisional_rows:
        status = "AI_PROVISIONAL_COMPLETE" if complete else "AI_PROVISIONAL_INCOMPLETE"
    else:
        status = "COMPLETE_VALID" if complete else "INCOMPLETE"
    return {
        "status": status,
        "reviewed": dict(reviewed), "totals": totals, "errors": errors,
        "ai_provisional_rows": ai_provisional_rows,
        "clinical_expert_complete": complete and ai_provisional_rows == 0,
        "freeze_authorized": False,
        "note": (
            "AI provisional review cannot substitute for clinical expert review. "
            "Validation never promotes rules; freeze is a separate audited step."
        ),
    }


def main() -> None:
    args = parse_args()
    result = validate(args.review_root)
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload)
    print(payload, end="")
    if result["status"] == "INVALID":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
