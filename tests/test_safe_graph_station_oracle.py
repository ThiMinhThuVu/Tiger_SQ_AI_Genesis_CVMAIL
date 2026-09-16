from scripts.evaluate_safe_graph_station_oracle import oracle_station_support


def test_oracle_station_uses_disjunctive_max_support():
    row = {
        "frame": "sample.png",
        "predicted_class_id": "3",
        "visible_station_targets": "6L|7L",
    }
    support = {("6L", 3): 0.2, ("7L", 3): 0.8}
    assert oracle_station_support(row, support) == 0.8
