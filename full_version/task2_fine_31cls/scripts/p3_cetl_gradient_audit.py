#!/usr/bin/env python3
"""Measure native-vs-CETL gradient scale without updating model weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from pathlib import Path

import torch
import yaml
from torch import Tensor
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from full_version.task2_fine_31cls.scripts.data_full import (  # noqa: E402
    build_manifest,
    load_manifest,
    records_for_manifest,
)
from scripts.train_mask2former import collate  # noqa: E402
from scripts.train_mask2former_improved import build_improved  # noqa: E402
from tiger_models.cetl_loss import cross_entropy_tversky_loss  # noqa: E402
from tiger_models.data import LabelMap, TigerMultitaskDataset  # noqa: E402


GROUP_PREFIXES = {
    "class_predictor": ("segmenter.class_predictor.",),
    "transformer_decoder": ("segmenter.model.transformer_module.",),
    "backbone_stage4": (
        "segmenter.model.pixel_level_module.encoder.swin.encoder.layers.3.",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--cetl-config", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--audit-seed", type=int, default=94017)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_records(records: list, count: int, seed: int) -> list:
    """Choose a deterministic, shuffled subset spanning the full train list."""
    if count <= 0:
        raise ValueError("count must be positive")
    if count > len(records):
        raise ValueError(f"requested {count} records but only {len(records)} are available")
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    return [records[index] for index in indices[:count]]


def parameter_groups(model: torch.nn.Module) -> tuple[list[Tensor], dict[str, list[int]]]:
    parameters: list[Tensor] = []
    parameter_index: dict[int, int] = {}
    groups: dict[str, list[int]] = {name: [] for name in GROUP_PREFIXES}
    for parameter_name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        matching_groups = [
            group_name
            for group_name, prefixes in GROUP_PREFIXES.items()
            if parameter_name.startswith(prefixes)
        ]
        if not matching_groups:
            continue
        identity = id(parameter)
        if identity not in parameter_index:
            parameter_index[identity] = len(parameters)
            parameters.append(parameter)
        for group_name in matching_groups:
            groups[group_name].append(parameter_index[identity])
    empty = [name for name, indices in groups.items() if not indices]
    if empty:
        raise RuntimeError(f"No trainable parameters found for groups: {empty}")
    return parameters, groups


def gradient_comparison(
    native_gradients: tuple[Tensor | None, ...],
    cetl_gradients: tuple[Tensor | None, ...],
    indices: list[int],
    cetl_weight: float,
) -> dict[str, float]:
    native_squared = 0.0
    cetl_squared = 0.0
    dot = 0.0
    active = 0
    for index in indices:
        native = native_gradients[index]
        cetl = cetl_gradients[index]
        if native is None and cetl is None:
            continue
        active += 1
        if native is not None:
            native_squared += float(native.detach().float().square().sum().cpu())
        if cetl is not None:
            cetl_squared += float(cetl.detach().float().square().sum().cpu())
        if native is not None and cetl is not None:
            dot += float((native.detach().float() * cetl.detach().float()).sum().cpu())
    native_norm = math.sqrt(native_squared)
    cetl_norm = math.sqrt(cetl_squared)
    denominator = native_norm * cetl_norm
    raw_ratio = cetl_norm / native_norm if native_norm > 0.0 else float("inf")
    return {
        "active_parameter_tensors": active,
        "native_gradient_norm": native_norm,
        "cetl_gradient_norm": cetl_norm,
        "raw_norm_ratio": raw_ratio,
        "effective_norm_ratio": cetl_weight * raw_ratio,
        "cosine_similarity": dot / denominator if denominator > 0.0 else 0.0,
    }


def summarize_batches(rows: list[dict], cetl_weight: float) -> dict:
    summaries = {}
    for group_name in GROUP_PREFIXES:
        group_rows = [row["gradient_groups"][group_name] for row in rows]
        summaries[group_name] = {
            key: statistics.median(float(row[key]) for row in group_rows)
            for key in (
                "native_gradient_norm",
                "cetl_gradient_norm",
                "raw_norm_ratio",
                "effective_norm_ratio",
                "cosine_similarity",
            )
        }
    effective = [
        summaries[name]["effective_norm_ratio"]
        for name in ("class_predictor", "transformer_decoder", "backbone_stage4")
    ]
    cosines = [
        summaries[name]["cosine_similarity"]
        for name in ("class_predictor", "transformer_decoder", "backbone_stage4")
    ]
    raw_ratios = [
        summaries[name]["raw_norm_ratio"]
        for name in ("class_predictor", "transformer_decoder", "backbone_stage4")
    ]
    underweighted = sum(value < 0.05 for value in effective) >= 2
    conflict = min(cosines) < -0.2
    target_ratio = 0.075
    recommended_weight = target_ratio / max(statistics.median(raw_ratios), 1e-12)
    recommended_weight = min(max(recommended_weight, 0.5), 4.0)
    if underweighted and not conflict:
        decision = "RUN_ONE_GRADIENT_CALIBRATED_CETL_VARIANT"
    else:
        decision = "STOP_CETL_AND_MOVE_TO_SAFE_TTA"
    return {
        "groups": summaries,
        "mechanical_gate": {
            "current_cetl_weight": cetl_weight,
            "underweighted_in_at_least_two_of_three_groups": underweighted,
            "any_group_cosine_below_minus_0_2": conflict,
            "target_effective_gradient_ratio": target_ratio,
            "recommended_cetl_weight_if_rescaled": recommended_weight,
            "decision": decision,
        },
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit: {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the gradient audit")
    if args.batches <= 0 or args.batch_size <= 0:
        raise ValueError("batches and batch-size must be positive")

    base_config = yaml.safe_load(args.base_config.read_text())
    cetl_config = yaml.safe_load(args.cetl_config.read_text())
    cetl_weight = float(cetl_config["cetl_aux_loss_weight"])
    manifest = load_manifest(args.split_manifest)
    current = build_manifest(args.data_root, int(base_config["split_seed"]))
    if manifest.get("snapshot") != current.get("snapshot"):
        raise ValueError("Current dataset does not match the immutable split manifest")
    if int(cetl_config["split_seed"]) != int(base_config["split_seed"]):
        raise ValueError("Base and CETL configs use different split seeds")

    checkpoint = args.checkpoint_root / f"fold_{args.fold}" / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    labelmap = LabelMap.load(args.data_root / "labelmap.csv")
    base_config["fine_to_coarse"] = labelmap.fine_to_coarse.astype(int).tolist()
    device = torch.device("cuda")
    model = build_improved(base_config, labelmap).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    model.train()

    train_records = records_for_manifest(args.data_root, manifest, args.fold, "train")
    selected_records = select_records(
        train_records,
        args.batches * args.batch_size,
        args.audit_seed + args.fold,
    )
    dataset = TigerMultitaskDataset(
        selected_records,
        labelmap,
        int(base_config["height"]),
        int(base_config["width"]),
        False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
        pin_memory=True,
    )
    parameters, groups = parameter_groups(model)
    class_weights = torch.from_numpy(labelmap.fine_weights).to(device)
    rows = []
    for batch_index, batch in enumerate(loader):
        batch_seed = args.audit_seed + 1000 * args.fold + batch_index
        random.seed(batch_seed)
        torch.manual_seed(batch_seed)
        torch.cuda.manual_seed_all(batch_seed)
        result = model(
            batch["image"].to(device),
            [item.to(device) for item in batch["mask_labels"]],
            [item.to(device) for item in batch["class_labels"]],
        )
        native_loss = result["output"].loss
        cetl_loss, cetl_terms = cross_entropy_tversky_loss(
            result["fine_probability"],
            batch["fine"].to(device),
            class_weights=class_weights,
            ce_fraction=float(cetl_config["cetl_ce_fraction"]),
            alpha=float(cetl_config["cetl_tversky_alpha"]),
            beta=float(cetl_config["cetl_tversky_beta"]),
            include_background=bool(cetl_config["cetl_include_background"]),
            present_classes_only=bool(cetl_config["cetl_present_classes_only"]),
            smooth=float(cetl_config["cetl_smooth"]),
        )
        native_gradients = torch.autograd.grad(
            native_loss, parameters, retain_graph=True, allow_unused=True
        )
        cetl_gradients = torch.autograd.grad(
            cetl_loss, parameters, allow_unused=True
        )
        row = {
            "batch": batch_index,
            "batch_seed": batch_seed,
            "names": list(batch["name"]),
            "native_loss": float(native_loss.detach().cpu()),
            "cetl_loss": float(cetl_loss.detach().cpu()),
            "cetl_cross_entropy": float(cetl_terms["cross_entropy"].detach().cpu()),
            "cetl_tversky": float(cetl_terms["tversky"].detach().cpu()),
            "gradient_groups": {
                name: gradient_comparison(
                    native_gradients, cetl_gradients, indices, cetl_weight
                )
                for name, indices in groups.items()
            },
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError("Audit unexpectedly populated parameter .grad buffers")

    output = {
        "material_passport": {
            "origin_skill": "experiment-agent",
            "origin_mode": "run",
            "origin_date": "2026-09-09",
            "verification_status": "ANALYZED",
            "version_label": "p3_cetl_gradient_audit_v1",
        },
        "experiment": "p3_cetl_gradient_audit",
        "fold": args.fold,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_epoch": int(payload["epoch"]),
        "base_config": str(args.base_config.resolve()),
        "cetl_config": str(args.cetl_config.resolve()),
        "split_manifest": str(args.split_manifest.resolve()),
        "audit_seed": args.audit_seed,
        "batches": args.batches,
        "batch_size": args.batch_size,
        "optimizer_steps": 0,
        "selected_names": [record.image.name for record in selected_records],
        "batch_results": rows,
        "summary": summarize_batches(rows, cetl_weight),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "summary": output["summary"]}, indent=2))


if __name__ == "__main__":
    main()
