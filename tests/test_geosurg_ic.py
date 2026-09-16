import torch

from tiger_models.geosurg_ic import appearance_views, interface_probe, mixed_response


def test_additive_context_is_not_penalized():
    base, a, b = torch.randn(3, 4), torch.randn(3, 4), torch.randn(3, 4)
    delta = mixed_response([base, base+a, base+b, base+a+b])
    assert torch.allclose(delta, torch.zeros_like(delta), atol=1e-6)


def test_interaction_and_replay_gradient():
    p = torch.tensor(0.7, requires_grad=True)
    def views():
        return [p**2, 2*p**2, 3*p**2, 7*p**2]
    loss = mixed_response(views()).square()
    expected, = torch.autograd.grad(loss, p)
    with torch.no_grad():
        delta = mixed_response(views())
    for sign, value in zip((1, -1, -1, 1), views()):
        (sign * 2 * delta * value).backward()
    assert torch.allclose(p.grad, expected)


def example():
    target = torch.ones(32, 40, dtype=torch.long)
    target[:, 20:] = 2
    depth = torch.zeros(32, 40)
    depth[:, 20:] = 1
    return target, depth


def test_routing_is_count_matched_and_inside_gt_interface():
    target, depth = example()
    probe = interface_probe(target, depth, 17)
    counts = [(w > 0).sum().item() for w in probe["weights"].values()]
    assert counts[0] > 0 and len(set(counts)) == 1
    for w in probe["weights"].values():
        assert torch.allclose(w.sum(), torch.tensor(1.))
        assert w[:, :18].sum() == 0 and w[:, 22:].sum() == 0


def test_flat_depth_disables_ic():
    target, depth = example()
    probe = interface_probe(target, depth * 0, 17)
    assert probe["pixels"] == 0


def test_appearance_probes_are_disjoint_and_factorial():
    target, depth = example()
    probe = interface_probe(target, depth, 17)
    x = torch.zeros(1, 3, 64, 80)
    views = appearance_views(x, probe, 91)
    assert torch.allclose(views[3], views[1] + views[2] - views[0])
    assert torch.all((views[1]-x) * (views[2]-x) == 0)
    assert torch.equal(views[0], x)


def test_no_boundary_is_supported():
    assert interface_probe(torch.ones(20, 20, dtype=torch.long), torch.rand(20, 20), 3) is None


def test_depth_affine_scale_does_not_change_routing():
    target, depth = example()
    p = interface_probe(target, depth, 17)
    q = interface_probe(target, depth * 4 + 7, 17)
    assert torch.equal(p["weights"]["geo"], q["weights"]["geo"])


def test_case_then_center_aggregation_not_frame_weighted():
    from full_version.geosurg_ic.evaluate_native import aggregate
    def row(case, score):
        return {"case": case, "dice": score, "nhd": 1-score, "score": score,
                "boundary_band_dice": score, "boundary_band_accuracy": score}
    result = aggregate([row("center_1_case_1", 0), row("center_1_case_1", 1),
                        row("center_1_case_2", 1), row("center_2_case_1", 0)])
    assert result["per_case"]["center_1_case_1"]["score"] == .5
    assert result["case_macro"]["score"] == .5
    assert result["center_macro"]["score"] == .375


def test_geometry_and_shuffle_keep_same_budget_but_change_locations():
    target, depth = example()
    depth[16:, 20:] *= .1
    p = interface_probe(target, depth, 31)
    assert not torch.equal(p["weights"]["geo"], p["weights"]["shuffle"])
    assert torch.equal((p["weights"]["geo"] > 0).sum(), (p["weights"]["shuffle"] > 0).sum())


def test_native_frame_metric_matches_existing_evaluator(tmp_path):
    import numpy as np
    from pathlib import Path
    from types import SimpleNamespace
    from PIL import Image
    from full_version.geosurg_ic.evaluate_native import score_frame
    from tiger_models.data import LabelMap
    from tiger_models.metrics import segmentation_metrics
    labelmap = LabelMap.load(Path(__file__).resolve().parents[1] / "data/labelmap.csv")
    target = np.ones((32, 50), dtype=np.uint8)
    target[:, 25:] = 2
    prediction = target.copy()
    prediction[:, 23:25] = 2
    mask = tmp_path / "mask.png"
    Image.fromarray(labelmap.encode_fine(target)).save(mask)
    record = SimpleNamespace(fine_mask=mask, name="synthetic", case_id="center_1_case_1")
    got = score_frame(record, prediction, labelmap)
    expected = segmentation_metrics([prediction], [target], [record.case_id],
                                    labelmap.fine_weights, labelmap.fine_names)
    assert abs(got["dice"] - expected["dice"]) < 1e-7
    assert abs(got["nhd"] - expected["nhd"]) < 1e-7
    perfect = score_frame(record, target, labelmap)
    assert perfect["dice"] == 1 and perfect["nhd"] == 0
    assert perfect["boundary_band_dice"] == 1
