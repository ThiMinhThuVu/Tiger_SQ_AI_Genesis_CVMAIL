#!/usr/bin/env python3
"""Fill the isolated SAFE-Graph pilot copy with conservative AI-only judgments.

This script is intentionally scoped to the copied ``*_ai_provisional`` review
package.  It does not promote knowledge-graph rules or authorize correction.
The component decisions below were recorded after visual inspection of all 44
montages on 2026-08-10.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REVIEW_ROOT = ROOT / (
    "artifacts/safe_graph_knowledge/kg_v1_20260808/"
    "clinical_pilot_v1_r2_ai_provisional"
)
REVIEWER = "AI_PROVISIONAL_AGENT"
REVIEW_DATE = "2026-08-10"


# valid, severity, warning usefulness, tool occlusion, camera ambiguity
COMPONENT_DECISIONS: dict[str, tuple[str, str, str, str, str]] = {
    "CE-001": ("YES", "2", "YES", "NO", "NO"),
    "CE-002": ("UNCERTAIN", "1", "UNCERTAIN", "NO", "UNCERTAIN"),
    "CE-003": ("UNCERTAIN", "1", "UNCERTAIN", "NO", "UNCERTAIN"),
    "CE-004": ("UNCERTAIN", "0", "NO", "NO", "UNCERTAIN"),
    "CE-005": ("YES", "2", "YES", "NO", "NO"),
    "CE-006": ("UNCERTAIN", "0", "NO", "UNCERTAIN", "UNCERTAIN"),
    "CE-007": ("UNCERTAIN", "0", "NO", "NO", "UNCERTAIN"),
    "CE-008": ("UNCERTAIN", "0", "NO", "NO", "UNCERTAIN"),
    "CE-009": ("YES", "2", "YES", "NO", "NO"),
    "CE-010": ("YES", "2", "YES", "NO", "NO"),
    "CE-011": ("YES", "1", "UNCERTAIN", "NO", "NO"),
    "CE-012": ("UNCERTAIN", "1", "UNCERTAIN", "NO", "UNCERTAIN"),
    "CE-013": ("UNCERTAIN", "1", "UNCERTAIN", "NO", "UNCERTAIN"),
    "CE-014": ("UNCERTAIN", "0", "NO", "NO", "UNCERTAIN"),
    "CE-015": ("UNCERTAIN", "0", "NO", "NO", "UNCERTAIN"),
    "CE-016": ("UNCERTAIN", "0", "NO", "NO", "UNCERTAIN"),
    "CE-017": ("YES", "2", "YES", "NO", "NO"),
    "CE-018": ("UNCERTAIN", "1", "UNCERTAIN", "NO", "UNCERTAIN"),
    "CE-019": ("YES", "1", "UNCERTAIN", "NO", "NO"),
    "CE-020": ("UNCERTAIN", "1", "UNCERTAIN", "NO", "UNCERTAIN"),
    "CE-021": ("YES", "1", "UNCERTAIN", "NO", "NO"),
    "CE-022": ("UNCERTAIN", "0", "NO", "YES", "UNCERTAIN"),
    "CE-023": ("YES", "3", "YES", "NO", "NO"),
    "CE-024": ("YES", "2", "YES", "NO", "NO"),
    "CE-025": ("YES", "1", "UNCERTAIN", "NO", "NO"),
    "CE-026": ("UNCERTAIN", "1", "UNCERTAIN", "NO", "UNCERTAIN"),
    "CE-027": ("NO", "0", "NO", "NO", "UNCERTAIN"),
    "CE-028": ("YES", "3", "YES", "NO", "NO"),
    "CE-029": ("NO", "0", "NO", "NO", "UNCERTAIN"),
    "CE-030": ("UNCERTAIN", "0", "NO", "NO", "UNCERTAIN"),
    "CE-031": ("NO", "0", "NO", "NO", "UNCERTAIN"),
    "CE-032": ("YES", "2", "YES", "NO", "NO"),
    "CE-033": ("YES", "1", "UNCERTAIN", "NO", "NO"),
    "CE-034": ("YES", "2", "YES", "NO", "NO"),
    "CE-035": ("YES", "3", "YES", "NO", "NO"),
    "CE-036": ("YES", "3", "YES", "NO", "NO"),
    "CE-037": ("YES", "2", "YES", "NO", "NO"),
    "CE-038": ("YES", "1", "UNCERTAIN", "NO", "NO"),
    "CE-039": ("YES", "3", "YES", "NO", "NO"),
    "CE-040": ("YES", "2", "YES", "NO", "NO"),
    "CE-041": ("YES", "2", "YES", "NO", "NO"),
    "CE-042": ("YES", "3", "YES", "NO", "NO"),
    "CE-043": ("YES", "3", "YES", "NO", "NO"),
    "CE-044": ("YES", "1", "UNCERTAIN", "NO", "NO"),
}


CONFUSION_DECISIONS = {
    "CF-001": ("DEPENDS", "DEPENDS", "DEPENDS", "DEPENDS"),
    "CF-003": ("DEPENDS", "DEPENDS", "DEPENDS", "DEPENDS"),
    "CF-004": ("DEPENDS", "DEPENDS", "DEPENDS", "DEPENDS"),
    "CF-005": ("VALID", "YES", "DEPENDS", "DEPENDS"),
    "CF-006": ("DEPENDS", "DEPENDS", "DEPENDS", "DEPENDS"),
}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def component_cue(row: dict[str, str]) -> str:
    family = row["automatic_error_label"]
    pred = row["predicted_class_name"]
    gt = row["dominant_gt_class_name"]
    if family == "CLASS_SWAP":
        return f"station-conditioned {pred} versus {gt} topology plus calibrated confidence"
    if family == "FRAGMENTED_PREDICTION":
        return "within-class continuity, component count, uncertainty, and occlusion-aware expected shape"
    if family == "BOUNDARY_OR_PARTIAL_ERROR":
        return "boundary consistency with adjacent structures, component overlap, and uncertainty"
    if family in {"BACKGROUND_HALLUCINATION", "ABSENT_CLASS_HALLUCINATION"}:
        return "confidence, station support, and alternative-class topology; absence must not be a hard rule"
    if family == "AMBIGUOUS_FALSE_POSITIVE":
        return f"component-level {pred} versus {gt} evidence and station-conditioned topology"
    return "soft station context, visible neighboring landmarks, uncertainty, and tool-occlusion gating"


def component_comment(row: dict[str, str], decision: str) -> str:
    area = int(row["area_pixels"])
    family = row["automatic_error_label"]
    if decision == "YES":
        reason = (
            f"montage shows a substantive prediction-GT discordance consistent with {family}; "
            "clinical severity and GT correctness still need clinical confirmation"
        )
    elif decision == "NO":
        reason = (
            f"the {area}-pixel discrepancy appears too small to count as a clinically meaningful error; "
            "this needs clinical confirmation"
        )
    else:
        reason = (
            f"montage evidence is insufficient to resolve the {family} label confidently, especially at "
            f"{row['area_bin']} component scale; this needs clinical confirmation"
        )
    return "AI_PROVISIONAL: " + reason + "."


def fill_components() -> Counter:
    path = REVIEW_ROOT / "component_error_review.csv"
    fields, rows = read_csv(path)
    assert {row["review_id"] for row in rows} == set(COMPONENT_DECISIONS)
    for row in rows:
        decision, severity, warning, tool, camera = COMPONENT_DECISIONS[row["review_id"]]
        row["clinical_error_valid_yes_no_uncertain"] = decision
        row["clinical_error_family"] = row["automatic_error_label"]
        row["clinical_severity_0_1_2_3"] = severity
        row["warning_would_be_useful_yes_no_uncertain"] = warning
        row["knowledge_cue"] = component_cue(row)
        row["tool_occluded_yes_no_uncertain"] = tool
        row["camera_ambiguous_yes_no_uncertain"] = camera
        row["reviewer_id"] = REVIEWER
        row["review_date"] = REVIEW_DATE
        row["review_comment"] = component_comment(row, decision)
    write_csv(path, fields, rows)
    return Counter(row["clinical_error_valid_yes_no_uncertain"] for row in rows)


def set_common_kg_fields(
    row: dict[str, str], *, decision: str, scope: str, warning: str,
    camera: str, tool: str, exception: str, comment: str,
) -> None:
    row["expert_decision_valid_invalid_depends"] = decision
    row["expert_relation_scope"] = scope
    row["expert_constraint_hard_soft"] = "SOFT"
    row["expert_absence_is_violation"] = "NO"
    row["expert_warning_eligible"] = warning
    row["expert_correction_eligible"] = "NO"
    row["expert_camera_dependency"] = camera
    row["expert_tool_occlusion_dependency"] = tool
    row["expert_exception"] = exception
    row["reviewer_id"] = REVIEWER
    row["review_date"] = REVIEW_DATE
    row["review_comment"] = "AI_PROVISIONAL: " + comment + "; needs clinical confirmation."


def fill_confusions() -> Counter:
    path = REVIEW_ROOT / "priority_confusion_review.csv"
    fields, rows = read_csv(path)
    for row in rows:
        rule = row["confusion_rule_id"]
        decision, warning, camera, tool = CONFUSION_DECISIONS[rule]
        names = f"{row['name_i']} versus {row['name_j']}"
        comment = (
            f"OOF provenance is preserved; {names} is treated only as a model-error hypothesis, "
            "not as an anatomical relation or a correction rule"
        )
        if rule == "CF-005":
            comment += "; multiple medium/large montage errors support provisional warning relevance"
        else:
            comment += "; available montage evidence is limited or dominated by small components"
        set_common_kg_fields(
            row, decision=decision,
            scope=f"OOF-validation component/pixel confusion for {names}; model-specific warning scope only",
            warning=warning, camera=camera, tool=tool,
            exception="Do not infer anatomical impossibility; abstain for poor exposure, occlusion, or uncertain alternatives",
            comment=comment,
        )
    write_csv(path, fields, rows)
    return Counter(row["expert_decision_valid_invalid_depends"] for row in rows)


def fill_relations() -> Counter:
    path = REVIEW_ROOT / "priority_anatomy_relation_review.csv"
    fields, rows = read_csv(path)
    for row in rows:
        protocol = row["candidate_type"] == "protocol_relation"
        pair = f"{row['subject']}--{row['object']}"
        station = row["station"]
        if protocol:
            decision = "DEPENDS"
            scope = (
                f"protocol-sourced {row['protocol_relation']} relation for co-visible {pair} in station {station}; "
                "single-frame 2D projection only"
            )
            comment = (
                "protocol provenance is preserved, but partial observability and zero/low GT co-visibility "
                "prevent AI-only confirmation as an operational rule"
            )
        else:
            decision = "DEPENDS"
            scope = (
                f"empirical-only station-{station} co-visible 2D relation for {pair}; descriptive dataset scope only"
            )
            comment = (
                "empirical-only provenance is preserved; frequent touching may reflect annotation style, "
                "station confounding, or projection rather than stable anatomy"
            )
        set_common_kg_fields(
            row, decision=decision, scope=scope, warning="DEPENDS",
            camera="YES", tool="DEPENDS",
            exception="Apply only when both anatomies are confidently visible; abstain under crop, deformation, or occlusion",
            comment=comment,
        )
    write_csv(path, fields, rows)
    return Counter(row["expert_decision_valid_invalid_depends"] for row in rows)


def fill_stations() -> Counter:
    path = REVIEW_ROOT / "priority_station_review.csv"
    fields, rows = read_csv(path)
    for row in rows:
        protocol = row["candidate_type"] == "protocol_edge"
        station = row["station"]
        anatomy = row["anatomy"]
        if protocol:
            decision = "DEPENDS"
            scope = (
                f"protocol-sourced station {station} boundary/context association for {anatomy}; "
                "visibility-conditioned single-frame warning scope"
            )
            comment = (
                "protocol provenance is preserved, but boundary membership does not imply mandatory visibility "
                f"and GT visibility probability is {row['gt_frame_probability']}"
            )
        else:
            decision = "DEPENDS"
            scope = (
                f"empirical-only station {station} co-visibility association for {anatomy}; "
                "descriptive dataset scope only"
            )
            comment = (
                "empirical-only provenance is preserved; observed frequency cannot establish anatomy inclusion, "
                "necessity, or impossibility"
            )
        set_common_kg_fields(
            row, decision=decision, scope=scope, warning="DEPENDS",
            camera="YES", tool="DEPENDS",
            exception="Absence is never a violation; abstain when station, exposure, field of view, or occlusion is uncertain",
            comment=comment,
        )
    write_csv(path, fields, rows)
    return Counter(row["expert_decision_valid_invalid_depends"] for row in rows)


def update_manifest(counts: dict[str, Counter]) -> None:
    path = REVIEW_ROOT / "manifest.json"
    data = json.loads(path.read_text())
    data.update({
        "version": "clinical_pilot_v1_r2_ai_provisional",
        "status": "AI_PROVISIONAL_COMPLETE_PENDING_CLINICAL_CONFIRMATION",
        "review_type": "AI_PROVISIONAL_NOT_CLINICAL_EXPERT",
        "reviewer_id": REVIEWER,
        "review_date": REVIEW_DATE,
        "freeze_authorized": False,
        "test_authorized": False,
        "correction_authorized": False,
        "ai_provisional_decision_counts": {
            key: dict(value) for key, value in counts.items()
        },
    })
    path.write_text(json.dumps(data, indent=2) + "\n")


def main() -> None:
    assert REVIEW_ROOT.is_dir(), REVIEW_ROOT
    counts = {
        "component": fill_components(),
        "confusion": fill_confusions(),
        "relation": fill_relations(),
        "station": fill_stations(),
    }
    update_manifest(counts)
    print(json.dumps({key: dict(value) for key, value in counts.items()}, indent=2))


if __name__ == "__main__":
    main()
