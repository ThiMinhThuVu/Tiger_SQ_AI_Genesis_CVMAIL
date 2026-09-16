#!/usr/bin/env python3
"""Export frozen DINOv2 ROI/ring evidence for the CF005 component cohort."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))
from scripts.evaluate_safe_graph_q3_ai_provisional import recover_global_components  # noqa: E402
from scripts.export_safe_graph_oof import validation_records_from_reference  # noqa: E402
from tiger_models.data import LabelMap, TigerMultitaskDataset  # noqa: E402


ENCODER_PREFIX = "encoder."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pair-cohort", type=Path,
        default=ROOT / "artifacts/safe_graph_verifier/p2_dense_cf005_v1/dense_pair_cohort.csv",
    )
    parser.add_argument(
        "--reference-oof-root", type=Path,
        default=ROOT / "artifacts/safe_graph_oof/improved_v2_validation_p2_dense_cf005_v1",
    )
    parser.add_argument(
        "--dinov2-root", type=Path,
        default=ROOT / "artifacts/new_baselines/dinov2_vits14",
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ring-pixels", type=int, default=24)
    parser.add_argument("--min-token-support", type=float, default=0.25)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def encoder_state(checkpoint: dict) -> dict[str, torch.Tensor]:
    return {
        key[len(ENCODER_PREFIX):]: value
        for key, value in checkpoint["model"].items()
        if key.startswith(ENCODER_PREFIX)
    }


def tensor_state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def verify_frozen_encoders(root: Path) -> tuple[dict, list[dict]]:
    records = []
    first_checkpoint = None
    hashes = set()
    for fold in range(5):
        path = root / f"fold_{fold}/best.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("config", {}).get("freeze_encoder") is not True:
            raise ValueError(f"DINOv2 encoder was not frozen in {path}")
        state = encoder_state(checkpoint)
        state_hash = tensor_state_sha256(state)
        hashes.add(state_hash)
        records.append({
            "fold": fold,
            "checkpoint": str(path),
            "checkpoint_sha256": file_sha256(path),
            "encoder_tensor_sha256": state_hash,
            "checkpoint_epoch": checkpoint.get("epoch"),
            "freeze_encoder": True,
        })
        if first_checkpoint is None:
            first_checkpoint = checkpoint
    if len(hashes) != 1:
        raise ValueError(f"Frozen DINOv2 encoders differ across folds: {sorted(hashes)}")
    return first_checkpoint, records


def weighted_mean(feature: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if feature.ndim != 3 or weights.shape != feature.shape[:2]:
        raise ValueError("Feature and weight shapes are incompatible")
    support = float(weights.sum())
    if support <= 0:
        raise ValueError("Weighted mean requires positive support")
    return (feature * weights[..., None]).sum((0, 1)) / support


def resize_mask_weights(mask: np.ndarray, grid_shape: tuple[int, int]) -> np.ndarray:
    tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    return F.interpolate(tensor, size=grid_shape, mode="area")[0, 0].numpy()


def bbox_mask(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if not len(ys):
        raise ValueError("Cannot construct bbox for empty mask")
    output = np.zeros_like(mask, dtype=bool)
    output[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
    return output


def component_geometry(mask: np.ndarray) -> dict[str, float]:
    ys, xs = np.nonzero(mask)
    area = float(mask.sum())
    height = float(ys.max() - ys.min() + 1)
    width = float(xs.max() - xs.min() + 1)
    boundary = mask & ~ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))
    perimeter = float(boundary.sum())
    return {
        "roi_area_fraction": area / mask.size,
        "roi_bbox_aspect": width / max(height, 1.0),
        "roi_bbox_fill": area / max(width * height, 1.0),
        "roi_centroid_x": float(xs.mean() / mask.shape[1]),
        "roi_centroid_y": float(ys.mean() / mask.shape[0]),
        "roi_compactness": float(4.0 * math.pi * area / max(perimeter * perimeter, 1.0)),
    }


def pool_component_embeddings(
    feature: np.ndarray,
    cls_embedding: np.ndarray,
    component: np.ndarray,
    ring_pixels: int,
    min_token_support: float,
) -> tuple[dict[str, np.ndarray], dict[str, float | bool]]:
    weights = resize_mask_weights(component, feature.shape[:2])
    raw_support = float(weights.sum())
    low_support = raw_support < min_token_support
    if low_support:
        weights = resize_mask_weights(bbox_mask(component), feature.shape[:2])
    roi = weighted_mean(feature, weights)

    distance = ndimage.distance_transform_edt(~component)
    ring = (distance > 0) & (distance <= ring_pixels)
    ring_weights = resize_mask_weights(ring, feature.shape[:2])
    ring_available = float(ring_weights.sum()) > 0
    ring_embedding = weighted_mean(feature, ring_weights) if ring_available else cls_embedding.copy()
    return {
        "roi": roi.astype(np.float32),
        "ring": ring_embedding.astype(np.float32),
        "contrast": (roi - ring_embedding).astype(np.float32),
        "global_cls": cls_embedding.astype(np.float32),
    }, {
        "roi_token_support": raw_support,
        "roi_low_token_support": bool(low_support),
        "roi_pooling_source": "bbox_fallback" if low_support else "predicted_mask",
        "ring_token_support": float(ring_weights.sum()),
        "ring_available": bool(ring_available),
    }


def build_encoder(checkpoint: dict, device: torch.device):
    from transformers import Dinov2Config, Dinov2Model

    checkpoint_name = checkpoint["config"]["pretrained_checkpoint"]
    config = Dinov2Config.from_pretrained(checkpoint_name, local_files_only=True)
    encoder = Dinov2Model(config)
    encoder.load_state_dict(encoder_state(checkpoint), strict=True)
    encoder.eval().to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    return encoder, checkpoint_name


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output root: {args.output_root}")
    if args.ring_pixels <= 0 or args.min_token_support <= 0:
        raise ValueError("Ring size and token support threshold must be positive")
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("Visual ROI export requires CUDA")

    pair_rows = read_csv(args.pair_cohort)
    if len(pair_rows) != 744:
        raise ValueError(f"Expected 744 CF005 components, got {len(pair_rows)}")
    pair_by_frame = defaultdict(list)
    for row in pair_rows:
        pair_by_frame[(int(row["fold"]), row["frame"])].append(row)

    checkpoint, encoder_records = verify_frozen_encoders(args.dinov2_root)
    encoder_hash = encoder_records[0]["encoder_tensor_sha256"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, checkpoint_name = build_encoder(checkpoint, device)
    labelmap = LabelMap.load(args.data_root / "labelmap.csv")

    feature_rows = []
    roi_embeddings = []
    ring_embeddings = []
    contrast_embeddings = []
    global_embeddings = []
    for fold in range(5):
        records = validation_records_from_reference(
            args.data_root, fold, args.reference_oof_root
        )
        dataset = TigerMultitaskDataset(records, labelmap, 512, 896, False)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
        for batch in loader:
            frame = batch["name"][0]
            selected = pair_by_frame.get((fold, frame), [])
            if not selected:
                continue
            output = encoder(pixel_values=batch["image"].to(device))
            tokens = output.last_hidden_state[0]
            patch = int(encoder.config.patch_size)
            grid_shape = (batch["image"].shape[-2] // patch, batch["image"].shape[-1] // patch)
            if tokens.shape[0] != 1 + grid_shape[0] * grid_shape[1]:
                raise ValueError(f"Unexpected DINO token grid for {frame}")
            cls_embedding = tokens[0].float().cpu().numpy()
            feature = tokens[1:].reshape(*grid_shape, -1).float().cpu().numpy()
            array_path = (
                args.reference_oof_root / f"fold_{fold}/frame_arrays/{Path(frame).stem}.npz"
            )
            with np.load(array_path) as arrays:
                prediction = arrays["prediction"]
                tool_probability = arrays["tool_probability"].astype(np.float32)
            components = recover_global_components(prediction)
            for row in selected:
                component_id = int(row["component_id"])
                component = components[component_id]
                if int(row["predicted_class_id"]) != int(prediction[component][0]):
                    raise ValueError(f"Component class mismatch for {frame}/{component_id}")
                embeddings, audit = pool_component_embeddings(
                    feature, cls_embedding, component,
                    args.ring_pixels, args.min_token_support,
                )
                row_index = len(feature_rows)
                roi_embeddings.append(embeddings["roi"])
                ring_embeddings.append(embeddings["ring"])
                contrast_embeddings.append(embeddings["contrast"])
                global_embeddings.append(embeddings["global_cls"])
                feature_rows.append({
                    "row_index": row_index,
                    "fold": fold,
                    "case_id": row["case_id"],
                    "frame": frame,
                    "component_id": component_id,
                    "predicted_class_id": row["predicted_class_id"],
                    "pair_swap_target": row["pair_swap_target"],
                    "dense_counterpart_probability": row["dense_counterpart_probability"],
                    "mean_tool_probability": row["mean_tool_probability"],
                    "roi_mean_tool_probability": float(tool_probability[component].mean()),
                    **component_geometry(component),
                    **audit,
                })

    if len(feature_rows) != len(pair_rows):
        raise ValueError(f"ROI export produced {len(feature_rows)} rows, expected {len(pair_rows)}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "component_index.csv", feature_rows)
    np.savez_compressed(
        args.output_root / "component_embeddings.npz",
        roi=np.asarray(roi_embeddings, dtype=np.float16),
        ring=np.asarray(ring_embeddings, dtype=np.float16),
        contrast=np.asarray(contrast_embeddings, dtype=np.float16),
        global_cls=np.asarray(global_embeddings, dtype=np.float16),
    )
    manifest = {
        "artifact": "CF005 frozen DINOv2 visual ROI features",
        "version": args.output_root.name,
        "status": "EXPORTED_UNEVALUATED",
        "components": len(feature_rows),
        "cases": sorted({row["case_id"] for row in feature_rows}),
        "source_pair_cohort": str(args.pair_cohort),
        "reference_oof_root": str(args.reference_oof_root),
        "encoder_checkpoint_name": checkpoint_name,
        "encoder_tensor_sha256": encoder_hash,
        "fold_encoder_audit": encoder_records,
        "encoder_frozen": True,
        "feature_input": "RGB frame + OOF predicted component mask only",
        "gt_used_as_feature": False,
        "test_data_used": False,
        "ring_pixels": args.ring_pixels,
        "min_token_support": args.min_token_support,
        "embedding_dimensions": int(np.asarray(roi_embeddings).shape[1]),
        "low_token_support_components": sum(
            bool(row["roi_low_token_support"]) for row in feature_rows
        ),
        "next_gate": "Run frozen V0-V7 LOCO specification from the information-gain plan",
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "output_root": str(args.output_root),
        "components": len(feature_rows),
        "encoder_tensor_sha256": encoder_hash,
        "low_token_support_components": manifest["low_token_support_components"],
        "test_data_used": False,
    }, indent=2))


if __name__ == "__main__":
    main()

