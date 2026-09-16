import json
from pathlib import Path

import numpy as np

from scripts.build_safe_graph_knowledge import (
    BinarySupport,
    empirical_status,
    mask_geometry,
    resize_ids,
)
from tiger_models.data import LabelMap


ROOT = Path(__file__).resolve().parents[1]


def test_corrected_pulmonary_ligament_laterality():
    seed = json.loads((ROOT / "knowledge_graph/clinical_station_anatomy_seed_v1.json").read_text())
    edges = {(item["station"], item["anatomy"]) for item in seed["station_anatomy_edges"]}
    assert ("13L", "Left inferior pulmonary ligament") in edges
    assert ("13R", "Right inferior pulmonary ligament") in edges
    assert ("13R", "Left inferior pulmonary ligament") not in edges
    assert ("13L", "Right inferior pulmonary ligament") not in edges


def test_seed_names_match_labelmap():
    seed = json.loads((ROOT / "knowledge_graph/clinical_station_anatomy_seed_v1.json").read_text())
    labelmap = LabelMap.load(ROOT / "data/labelmap.csv")
    names = set(labelmap.fine_names)
    assert all(item["anatomy"] in names for item in seed["station_anatomy_edges"])


def test_empirical_zero_is_never_forbidden():
    assert empirical_status(7, 0, 0.0) == "unobserved_not_forbidden"
    support = BinarySupport()
    for index in range(7):
        support.observe(f"case_{index}", False)
    assert support.summary()["empirical_status"] == "unobserved_not_forbidden"


def test_geometry_direction_and_touch():
    ids = np.zeros((20, 20), dtype=np.uint8)
    ids[5:10, 3:7] = 3
    ids[5:10, 8:12] = 4
    relation = mask_geometry(ids, [3, 4], touch_pixels=2.0)[(3, 4)]
    assert relation["centroid_dx_norm"] > 0
    assert abs(relation["centroid_dy_norm"]) < 1e-8
    assert relation["touch"] is True


def test_resize_preserves_discrete_ids():
    ids = np.zeros((100, 200), dtype=np.uint8)
    ids[:, 100:] = 13
    resized = resize_ids(ids, 50)
    assert resized.shape == (25, 50)
    assert set(np.unique(resized)) == {0, 13}


def test_anatomy_relation_seed_uses_canonical_labels():
    seed = json.loads((ROOT / "knowledge_graph/clinical_anatomy_relations_seed_v1.json").read_text())
    labelmap = LabelMap.load(ROOT / "data/labelmap.csv")
    names = set(labelmap.fine_names)
    assert seed["relations"]
    assert all(item["subject"] in names and item["object"] in names for item in seed["relations"])


def test_anatomy_relation_seed_cannot_auto_promote():
    seed = json.loads((ROOT / "knowledge_graph/clinical_anatomy_relations_seed_v1.json").read_text())
    assert seed["defaults"]["warning_eligible"] is False
    assert seed["defaults"]["correction_eligible"] is False
    assert seed["defaults"]["absence_is_violation"] is False
