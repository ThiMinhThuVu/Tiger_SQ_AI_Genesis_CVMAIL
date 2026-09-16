import numpy as np

from scripts.export_safe_graph_visual_roi_features import (
    component_geometry,
    pool_component_embeddings,
    tensor_state_sha256,
    weighted_mean,
)


def test_weighted_mean_uses_spatial_weights():
    feature = np.asarray([[[1.0, 0.0], [3.0, 2.0]]], dtype=np.float32)
    weights = np.asarray([[1.0, 0.0]], dtype=np.float32)
    assert np.allclose(weighted_mean(feature, weights), [1.0, 0.0])


def test_pool_component_embeddings_emits_fallback_for_tiny_mask():
    feature = np.arange(4 * 4 * 3, dtype=np.float32).reshape(4, 4, 3)
    component = np.zeros((64, 64), dtype=bool)
    component[3, 3] = True
    embeddings, audit = pool_component_embeddings(
        feature, np.zeros(3, dtype=np.float32), component,
        ring_pixels=4, min_token_support=0.25,
    )
    assert audit["roi_low_token_support"] is True
    assert audit["roi_pooling_source"] == "bbox_fallback"
    assert embeddings["roi"].shape == (3,)
    assert embeddings["contrast"].shape == (3,)


def test_component_geometry_is_finite():
    component = np.zeros((16, 20), dtype=bool)
    component[4:8, 5:11] = True
    result = component_geometry(component)
    assert set(result) == {
        "roi_area_fraction", "roi_bbox_aspect", "roi_bbox_fill",
        "roi_centroid_x", "roi_centroid_y", "roi_compactness",
    }
    assert all(np.isfinite(value) for value in result.values())


def test_tensor_state_hash_is_order_invariant():
    import torch
    first = {"b": torch.tensor([2]), "a": torch.tensor([1])}
    second = {"a": torch.tensor([1]), "b": torch.tensor([2])}
    assert tensor_state_sha256(first) == tensor_state_sha256(second)

