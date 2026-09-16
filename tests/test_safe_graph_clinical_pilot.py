import numpy as np

from scripts.prepare_safe_graph_clinical_pilot import (
    area_bin,
    diverse_select,
    global_component_mask,
    select_priority_tables,
)
from scripts.validate_safe_graph_clinical_review import (
    validate_component_review,
    validate_kg_review,
)


def test_global_component_mask_matches_classwise_global_order():
    mask = np.zeros((8, 10), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    mask[5:7, 1:3] = 1
    mask[2:5, 6:9] = 3
    recovered = global_component_mask(mask, class_id=3, component_id=3)
    assert recovered.sum() == 9
    assert recovered[3, 7]


def test_area_bins_are_frozen():
    assert area_bin(0.0001) == "tiny"
    assert area_bin(0.001) == "small"
    assert area_bin(0.005) == "medium"
    assert area_bin(0.02) == "large"


def test_diverse_select_spans_cases_before_repeating_group():
    rows = [
        {"case_id": "a", "frame": "a.png", "area_pixels": "100", "area_fraction": "0.01"},
        {"case_id": "a", "frame": "b.png", "area_pixels": "90", "area_fraction": "0.01"},
        {"case_id": "b", "frame": "c.png", "area_pixels": "80", "area_fraction": "0.01"},
    ]
    picked = diverse_select(rows, 2)
    assert {row["case_id"] for row in picked} == {"a", "b"}


def test_priority_tables_keep_p0_and_selected_pair_classes():
    station = [
        {"review_priority": "P2", "anatomy_id": "3"},
        {"review_priority": "P0", "anatomy_id": "20"},
    ]
    relation = [
        {"review_priority": "P1", "subject_id": "3", "object_id": "27"},
        {"review_priority": "P2", "subject_id": "8", "object_id": "9"},
    ]
    confusion = [{
        "review_priority": "P0", "class_i": "3", "class_j": "27", "edge_weight": "0.8"
    }]
    selected_station, selected_relation, selected_confusion = select_priority_tables(
        station, relation, confusion, 1
    )
    assert len(selected_station) == 2
    assert len(selected_relation) == 1
    assert len(selected_confusion) == 1


def test_review_validator_rejects_invalid_choice():
    row = {
        "review_id": "CE-001",
        "clinical_error_valid_yes_no_uncertain": "MAYBE",
        "clinical_severity_0_1_2_3": "",
        "warning_would_be_useful_yes_no_uncertain": "",
        "tool_occluded_yes_no_uncertain": "",
        "camera_ambiguous_yes_no_uncertain": "",
        "reviewer_id": "",
        "review_date": "",
    }
    errors = validate_component_review([row])
    assert any("invalid clinical_error" in error for error in errors)


def test_ai_provisional_kg_review_is_forced_soft_and_noncorrecting():
    row = {
        "rule_id": "SA-001",
        "expert_decision_valid_invalid_depends": "DEPENDS",
        "expert_relation_scope": "station-conditioned",
        "expert_constraint_hard_soft": "HARD",
        "expert_absence_is_violation": "YES",
        "expert_warning_eligible": "DEPENDS",
        "expert_correction_eligible": "YES",
        "expert_camera_dependency": "DEPENDS",
        "expert_tool_occlusion_dependency": "DEPENDS",
        "expert_exception": "occlusion",
        "reviewer_id": "AI_PROVISIONAL_AGENT",
        "review_date": "2026-08-10",
        "review_comment": "AI_PROVISIONAL: pending clinician",
    }
    errors = validate_kg_review([row], "rule_id")
    assert any("constraint must be SOFT" in error for error in errors)
    assert any("absence_is_violation must be NO" in error for error in errors)
    assert any("correction_eligible must be NO" in error for error in errors)
