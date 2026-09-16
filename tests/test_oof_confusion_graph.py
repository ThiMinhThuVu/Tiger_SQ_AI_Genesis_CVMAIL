from __future__ import annotations

import numpy as np

from scripts.build_oof_confusion_graph import build_graph
from tiger_models.data import LabelMap


def test_confusion_graph_is_test_free_aggregation():
    labelmap=LabelMap.load("data/labelmap.csv")
    matrix=np.eye(31,dtype=int)*10; matrix[19,7]=5; matrix[7,19]=2
    folds=[{"failure_metrics":{"confusion_pixels":matrix.tolist()}} for _ in range(5)]
    graph=build_graph(folds,labelmap,top_k=1); edge=graph["edges"][0]
    assert {edge["class_i"],edge["class_j"]}=={7,19}
    assert edge["i_to_j_pixels"]+edge["j_to_i_pixels"]==35
    assert edge["clinical_risk"]==3
