#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tiger_models import MultitaskSAM
from tiger_models.data import (
    STATIONS, LabelMap, TigerMultitaskDataset, audit_dataset, records_for_fold,
    visibility_pos_weight,
)
from tiger_models.losses import PrimarySAMLoss
from tiger_models.metrics import segmentation_metrics, selection_score, visibility_metrics
from tiger_models.runtime import (
    atomic_json, environment_info, git_state, load_checkpoint, save_checkpoint,
    seed_everything, seed_worker,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train prompt-free SAM 2.1 TIGER multi-task baseline")
    parser.add_argument("--model", choices=["sam2_hiera_tiny", "sam2_hiera_small", "sam2_hiera_base_plus"], default="sam2_hiera_tiny")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--finetune", choices=["frozen", "partial", "full"], default="partial")
    parser.add_argument("--mixed-precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--decoder-lr", type=float, default=3e-4)
    parser.add_argument("--visibility-head-lr", type=float, default=3e-4)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--boundary-loss-weight", type=float, default=0.20)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-kind", choices=["smoke", "stability", "primary"], default="primary")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true", help="engineering-only; never a primary benchmark")
    parser.add_argument("--stop-after-epoch", type=int, default=None,
                        help="engineering resume test: stop after this many completed epochs")
    return parser.parse_args()


def ensure_protocol(args: argparse.Namespace) -> None:
    if args.n_folds != 5 or args.fold not in range(5):
        raise ValueError("The fixed protocol requires fold 0..4 of exactly five folds")
    if (args.height, args.width) != (512, 896) and args.run_kind == "primary":
        raise ValueError("Primary runs must use 512x896")
    if args.batch_size * args.gradient_accumulation != 4:
        raise ValueError("Effective batch size must be exactly 4")
    if args.epochs != 100 and args.run_kind == "primary":
        raise ValueError("Primary runs require exactly 100 epochs")
    if args.run_kind == "smoke" and args.epochs != 1:
        raise ValueError("Smoke run must be exactly one epoch")
    if args.run_kind == "stability" and args.epochs != 10:
        raise ValueError("Stability run must be exactly ten epochs")
    if args.boundary_loss_weight <= 0:
        raise ValueError("Boundary-aware loss must have a positive weight")
    if args.early_stopping_patience < 1 or args.early_stopping_min_delta < 0:
        raise ValueError("Invalid early-stopping configuration")
    if args.stop_after_epoch is not None:
        if args.run_kind == "primary":
            raise ValueError("Early stop is not allowed for primary runs")
        if not 1 <= args.stop_after_epoch < args.epochs:
            raise ValueError("--stop-after-epoch must be between 1 and epochs-1")


def precision_for_device(requested: str, device: torch.device) -> tuple[str, str]:
    if device.type != "cuda":
        return "fp32", "CUDA unavailable; CPU mode is engineering-only"
    if requested == "bf16" and torch.cuda.is_bf16_supported():
        return "bf16", "requested BF16 is supported by the runtime and GPU"
    if requested == "bf16":
        return "fp16", "requested BF16 is unsupported; protocol fallback to FP16 GradScaler"
    return requested, "requested precision selected"


def autocast_context(device: torch.device, precision: str):
    if device.type == "cuda" and precision in {"bf16", "fp16"}:
        dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        return torch.autocast("cuda", dtype=dtype)
    return nullcontext()


def finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"Non-finite {name} detected")


def csv_write(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def csv_append(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    precision: str,
    labelmap: LabelMap,
    criterion: PrimarySAMLoss,
) -> tuple[dict, dict, float]:
    model.eval()
    fine_predictions: list[np.ndarray] = []
    fine_targets: list[np.ndarray] = []
    case_ids: list[str] = []
    names: list[str] = []
    visibility_targets: list[np.ndarray] = []
    visibility_probabilities: list[np.ndarray] = []
    inference_seconds = 0.0
    loss_totals: dict[str, float] = {}
    fine_correct = fine_pixels = visibility_correct = visibility_labels = 0
    if device.type == "cuda":
        torch.cuda.synchronize()
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        start = time.perf_counter()
        with autocast_context(device, precision):
            output = model(image)
            fine_target = batch["fine"].to(device, non_blocking=True)
            visibility_target = batch["visibility"].to(device, non_blocking=True)
            losses = criterion(output, fine_target, visibility_target)
        if device.type == "cuda":
            torch.cuda.synchronize()
        inference_seconds += time.perf_counter() - start
        finite("validation fine logits", output["fine_logits"])
        finite("validation visibility logits", output["visibility_logits"])
        probabilities = output["visibility_logits"].float().sigmoid()
        finite("validation probabilities", probabilities)
        fine_predictions.extend(output["fine_logits"].argmax(1).cpu().numpy())
        fine_targets.extend(batch["fine"].numpy())
        visibility_targets.extend(batch["visibility"].numpy())
        visibility_probabilities.extend(probabilities.cpu().numpy())
        case_ids.extend(batch["case_id"])
        names.extend(batch["name"])
        for key, value in losses.items():
            loss_totals[key] = loss_totals.get(key, 0.0) + float(value.detach())
        fine_prediction = output["fine_logits"].argmax(1)
        fine_correct += int((fine_prediction == fine_target).sum())
        fine_pixels += fine_target.numel()
        visibility_correct += int(((probabilities >= 0.5) == (visibility_target >= 0.5)).sum())
        visibility_labels += visibility_target.numel()

    fine = segmentation_metrics(
        fine_predictions, fine_targets, case_ids, labelmap.fine_weights, labelmap.fine_names
    )
    coarse_predictions = [labelmap.fine_to_coarse[prediction] for prediction in fine_predictions]
    coarse_targets = [labelmap.fine_to_coarse[target] for target in fine_targets]
    coarse = segmentation_metrics(
        coarse_predictions, coarse_targets, case_ids, labelmap.coarse_weights, labelmap.coarse_names
    )
    visibility_target = np.asarray(visibility_targets)
    visibility_probability = np.asarray(visibility_probabilities)
    visibility = visibility_metrics(visibility_target, visibility_probability, STATIONS)
    score = selection_score(
        fine["dice"], fine["nhd"], coarse["dice"], coarse["nhd"],
        visibility["macro_f1"], visibility["macro_auroc"],
    )
    metrics = {
        **{f"val_{key}": value / len(loader) for key, value in loss_totals.items()},
        "val_fine_pixel_accuracy": fine_correct / fine_pixels,
        "val_visibility_accuracy": visibility_correct / visibility_labels,
        "fine_dice": fine["dice"], "fine_nhd": fine["nhd"],
        "coarse_dice": coarse["dice"], "coarse_nhd": coarse["nhd"],
        "visibility_macro_f1": visibility["macro_f1"],
        "visibility_macro_auroc": visibility["macro_auroc"],
        "selection_score": score,
    }
    metrics["val_coarse_pixel_accuracy"] = float(np.mean([
        (prediction == target).mean()
        for prediction, target in zip(coarse_predictions, coarse_targets)
    ]))
    details = {
        "fine": fine, "coarse": coarse, "visibility": visibility,
        "names": names, "visibility_target": visibility_target,
        "visibility_probability": visibility_probability,
    }
    return metrics, details, 1000.0 * inference_seconds / len(loader.dataset)


def save_best_details(output_dir: Path, metrics: dict, details: dict, epoch: int) -> None:
    best = dict(metrics)
    best["best_epoch"] = epoch
    atomic_json(output_dir / "best_metrics.json", best)
    (output_dir / "best_epoch").write_text(f"{epoch}\n")
    csv_write(output_dir / "per_class_metrics.csv", [
        {"task": task, **row}
        for task in ("fine", "coarse") for row in details[task]["per_class"]
    ])
    csv_write(output_dir / "per_station_metrics.csv", details["visibility"]["per_station"])
    cases = sorted(details["fine"]["per_case_dice"])
    csv_write(output_dir / "per_case_metrics.csv", [{
        "case_id": case,
        "fine_dice": details["fine"]["per_case_dice"][case],
        "fine_nhd": details["fine"]["per_case_nhd"][case],
        "coarse_dice": details["coarse"]["per_case_dice"][case],
        "coarse_nhd": details["coarse"]["per_case_nhd"][case],
    } for case in cases])
    np.savez_compressed(
        output_dir / "validation_probabilities.npz",
        names=np.asarray(details["names"]),
        visibility_target=details["visibility_target"],
        visibility_probability=details["visibility_probability"],
    )


def main() -> int:
    args = parse_args()
    model_defaults = {
        "sam2_hiera_tiny": ("checkpoints/sam2.1_hiera_tiny.pt", "configs/sam2.1/sam2.1_hiera_t.yaml"),
        "sam2_hiera_small": ("checkpoints/sam2.1_hiera_small.pt", "configs/sam2.1/sam2.1_hiera_s.yaml"),
        "sam2_hiera_base_plus": ("checkpoints/sam2.1_hiera_base_plus.pt", "configs/sam2.1/sam2.1_hiera_b+.yaml"),
    }
    default_checkpoint, args.sam2_config = model_defaults[args.model]
    if args.checkpoint is None:
        args.checkpoint = ROOT / default_checkpoint
    ensure_protocol(args)
    seed_everything(args.seed)
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is unavailable. Submit this command to an RTX A5000 node.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and torch.cuda.device_count() != 1:
        # Visibility of multiple devices is treated as a configuration error; the job
        # must expose only one GPU rather than accidentally using shared resources.
        raise RuntimeError(f"Expected exactly one visible GPU, got {torch.cuda.device_count()}")
    if device.type == "cuda" and "RTX A5000" not in torch.cuda.get_device_name(0):
        raise RuntimeError(
            f"Primary resource protocol requires NVIDIA RTX A5000, got {torch.cuda.get_device_name(0)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    last_path = args.output_dir / "last.pt"
    if last_path.exists() and not args.resume:
        raise FileExistsError(f"{last_path} exists; pass --resume to continue without overwrite")
    if args.resume and not last_path.is_file():
        raise FileNotFoundError(f"Resume requested but {last_path} is absent")

    audit = audit_dataset(args.data_root, full=False)
    atomic_json(args.output_dir / "data_gate.json", audit)
    nvidia_smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
    (args.output_dir / "nvidia_smi.txt").write_text(nvidia_smi.stdout + nvidia_smi.stderr)
    if device.type == "cuda" and nvidia_smi.returncode != 0:
        raise RuntimeError("nvidia-smi failed on the CUDA execution node")

    labelmap = LabelMap.load(args.data_root / "labelmap.csv")
    train_records = records_for_fold(args.data_root, args.fold, "train")
    val_records = records_for_fold(args.data_root, args.fold, "validation")
    test_records = records_for_fold(args.data_root, args.fold, "test")
    split_case_sets = [{r.case_id for r in records} for records in (train_records, val_records, test_records)]
    if any(split_case_sets[i] & split_case_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("Case leakage detected immediately before training")
    train_dataset = TigerMultitaskDataset(train_records, labelmap, args.height, args.width, training=True)
    val_dataset = TigerMultitaskDataset(val_records, labelmap, args.height, args.width, training=False)
    generator = torch.Generator().manual_seed(args.seed)
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
    }
    if args.num_workers:
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator,
        drop_last=False, **loader_kwargs,
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, **loader_kwargs)
    expected_optimizer_steps = math.ceil(len(train_loader) / args.gradient_accumulation)
    if len(train_dataset) != 98 or len(val_dataset) != 14 or len(test_records) != 28 or expected_optimizer_steps != 25:
        raise RuntimeError(
            "Protocol budget changed: "
            f"train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_records)}, "
            f"optimizer_steps={expected_optimizer_steps}"
        )

    precision, precision_reason = precision_for_device(args.mixed_precision, device)
    model = MultitaskSAM(
        checkpoint_path=str(args.checkpoint), finetune=args.finetune,
        gradient_checkpointing=args.gradient_checkpointing, coarse_head=False,
        config_name=args.sam2_config,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameter_groups(args.encoder_lr, args.decoder_lr, args.visibility_head_lr),
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=6e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and precision == "fp16")
    criterion = PrimarySAMLoss(
        torch.from_numpy(labelmap.fine_weights),
        torch.from_numpy(visibility_pos_weight(train_records)),
        boundary_weight=args.boundary_loss_weight,
    ).to(device)

    counts = model.parameter_counts()
    freeze_policy = model.encoder.freeze_policy() if hasattr(model.encoder, "freeze_policy") else {}
    config = vars(args).copy()
    config = {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}
    config.update({
        "experiment_id": f"{args.model}_partial", "effective_batch_size": 4,
        "optimizer_steps_per_epoch": expected_optimizer_steps, "train_samples_per_epoch": 98,
        "validation_samples": 14, "test_samples": 28,
        "split_case_counts": {"train": 7, "validation": 1, "test": 2},
        "split_cases": {
            "train": sorted(split_case_sets[0]), "validation": sorted(split_case_sets[1]),
            "test": sorted(split_case_sets[2]),
        },
        "final_accumulation_window": len(train_loader) % args.gradient_accumulation,
        "precision_selected": precision,
        "precision_reason": precision_reason, "fine_to_coarse_strategy": "derived_from_fine",
        "coarse_auxiliary_head": False, "boundary_head": False,
        "boundary_aware_loss": "semantic edge weighted BCE + Dice", "oracle_prompt": False,
        "parameter_counts": counts, "freeze_policy": freeze_policy,
        "visibility_pos_weight": visibility_pos_weight(train_records).tolist(),
    })
    atomic_json(args.output_dir / "config.json", config)
    atomic_json(args.output_dir / "environment.json", environment_info())
    atomic_json(args.output_dir / "git_state.json", git_state(ROOT))
    (args.output_dir / "command.txt").write_text(" ".join([sys.executable, *sys.argv]) + "\n")

    resource_profile = {
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "gpu_count": torch.cuda.device_count() if device.type == "cuda" else 0,
        "peak_allocated_vram_gb": None, "peak_reserved_vram_gb": None,
        "physical_batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "effective_batch_size": 4, "final_effective_batch_size": 2,
        "mixed_precision": precision,
        "mixed_precision_reason": precision_reason,
        "gradient_checkpointing": args.gradient_checkpointing,
        "train_seconds_per_epoch": None, "validation_seconds": None,
        "inference_ms_per_image": None,
    }
    atomic_json(args.output_dir / "resource_profile.json", resource_profile)

    start_epoch, history = 0, []
    best_score, best_train_loss, best_val_loss = -math.inf, math.inf, math.inf
    early_stopping_bad_epochs = 0
    if args.resume:
        payload = load_checkpoint(last_path, model, optimizer, scheduler, scaler, device)
        if payload.get("data_generator_state") is not None:
            generator.set_state(payload["data_generator_state"].cpu())
        start_epoch = int(payload["epoch"]) + 1
        history = list(payload.get("history", []))
        previous_scores = [row.get("selection_score") for row in history]
        best_score = max((score for score in previous_scores if score is not None), default=-math.inf)
        best_train_loss = min((row["train_loss"] for row in history), default=math.inf)
        best_val_loss = min((row["val_loss"] for row in history), default=math.inf)
        early_stopping_bad_epochs = int(history[-1].get("early_stopping_bad_epochs", 0)) if history else 0
    if start_epoch >= args.epochs:
        raise RuntimeError(f"Checkpoint already reached epoch {start_epoch}; requested epochs={args.epochs}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    epoch_train_seconds: list[float] = []
    validation_seconds: list[float] = []
    iteration_seconds: list[float] = []
    final_epoch_exclusive = min(args.epochs, args.stop_after_epoch or args.epochs)
    stopped_early = False
    completed_epochs = start_epoch
    for epoch in range(start_epoch, final_epoch_exclusive):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running: dict[str, float] = {}
        train_fine_correct = train_fine_pixels = 0
        train_visibility_correct = train_visibility_labels = 0
        optimizer_steps = 0
        gradient_norms: list[float] = []
        batch_rows: list[dict] = []
        epoch_start = time.perf_counter()
        for step, batch in enumerate(train_loader):
            iteration_start = time.perf_counter()
            image = batch["image"].to(device, non_blocking=True)
            fine_target = batch["fine"].to(device, non_blocking=True)
            visibility_target = batch["visibility"].to(device, non_blocking=True)
            with autocast_context(device, precision):
                output = model(image)
                losses = criterion(output, fine_target, visibility_target)
                window_start = (step // args.gradient_accumulation) * args.gradient_accumulation
                accumulation_divisor = min(
                    args.gradient_accumulation, len(train_loader) - window_start
                )
                scaled_loss = losses["loss"] / accumulation_divisor
            finite("loss", losses["loss"])
            finite("training fine logits", output["fine_logits"])
            finite("training visibility logits", output["visibility_logits"])
            finite("training probabilities", output["visibility_logits"].float().sigmoid())
            scaler.scale(scaled_loss).backward()
            detached_losses = {key: float(value.detach()) for key, value in losses.items()}
            for key, value in detached_losses.items():
                running[key] = running.get(key, 0.0) + value
            fine_prediction = output["fine_logits"].argmax(1)
            visibility_prediction = output["visibility_logits"].sigmoid() >= 0.5
            batch_fine_correct = int((fine_prediction == fine_target).sum())
            batch_fine_pixels = fine_target.numel()
            batch_visibility_correct = int((visibility_prediction == (visibility_target >= 0.5)).sum())
            batch_visibility_labels = visibility_target.numel()
            train_fine_correct += batch_fine_correct
            train_fine_pixels += batch_fine_pixels
            train_visibility_correct += batch_visibility_correct
            train_visibility_labels += batch_visibility_labels
            batch_rows.append({
                "epoch": epoch, "batch": step, **detached_losses,
                "fine_pixel_accuracy": batch_fine_correct / batch_fine_pixels,
                "visibility_accuracy": batch_visibility_correct / batch_visibility_labels,
                "accumulation_divisor": accumulation_divisor,
            })
            accumulation_boundary = (
                (step + 1) % args.gradient_accumulation == 0 or step + 1 == len(train_loader)
            )
            if accumulation_boundary:
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
                finite("gradient norm", torch.as_tensor(gradient_norm))
                gradient_norms.append(float(gradient_norm))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
            if device.type == "cuda":
                torch.cuda.synchronize()
            iteration_seconds.append(time.perf_counter() - iteration_start)
        train_seconds = time.perf_counter() - epoch_start
        if optimizer_steps != expected_optimizer_steps:
            raise RuntimeError(
                f"Epoch {epoch} made {optimizer_steps} optimizer steps, expected {expected_optimizer_steps}"
            )
        epoch_train_seconds.append(train_seconds)
        csv_append(args.output_dir / "train_batch_metrics.csv", batch_rows)

        validation_start = time.perf_counter()
        metrics, details, inference_ms = validate(
            model, val_loader, device, precision, labelmap, criterion
        )
        validation_seconds.append(time.perf_counter() - validation_start)
        epoch_record = {
            "epoch": epoch,
            **{f"train_{key}": value / len(train_loader) for key, value in running.items()},
            "train_fine_pixel_accuracy": train_fine_correct / train_fine_pixels,
            "train_visibility_accuracy": train_visibility_correct / train_visibility_labels,
            "gradient_norm_mean": float(np.mean(gradient_norms)),
            "gradient_norm_max": float(np.max(gradient_norms)),
            "optimizer_steps": optimizer_steps,
            "train_seconds": train_seconds,
            "validation_seconds": validation_seconds[-1],
            "inference_ms_per_image": inference_ms,
            "learning_rates": {group.get("name", str(i)): group["lr"] for i, group in enumerate(optimizer.param_groups)},
            **metrics,
        }
        improved_val_loss = epoch_record["val_loss"] < (
            best_val_loss - args.early_stopping_min_delta
        )
        early_stopping_bad_epochs = 0 if improved_val_loss else early_stopping_bad_epochs + 1
        epoch_record["early_stopping_bad_epochs"] = early_stopping_bad_epochs
        history.append(epoch_record)
        atomic_json(args.output_dir / "history.json", history)
        csv_write(args.output_dir / "epoch_metrics.csv", history)
        print(json.dumps(epoch_record, sort_keys=True), flush=True)

        # Persist scheduler state for the *next* epoch so resume is exact.
        scheduler.step()
        score = metrics["selection_score"]
        if epoch_record["train_loss"] < best_train_loss:
            best_train_loss = epoch_record["train_loss"]
            save_checkpoint(
                args.output_dir / "best_train.pt", model, optimizer, scheduler, scaler,
                epoch, config, epoch_record, history, generator.get_state(),
            )
            atomic_json(args.output_dir / "best_train_metrics.json", {
                "criterion": "minimum_train_loss", "best_epoch": epoch,
                "best_train_loss": best_train_loss, **epoch_record,
            })
        if improved_val_loss:
            best_val_loss = epoch_record["val_loss"]
            save_checkpoint(
                args.output_dir / "best_val.pt", model, optimizer, scheduler, scaler,
                epoch, config, epoch_record, history, generator.get_state(),
            )
            atomic_json(args.output_dir / "best_val_metrics.json", {
                "criterion": "minimum_validation_loss", "best_epoch": epoch,
                "best_val_loss": best_val_loss, **epoch_record,
            })
        if score is not None and score > best_score:
            best_score = score
            save_checkpoint(
                args.output_dir / "best_selection.pt", model, optimizer, scheduler, scaler,
                epoch, config, metrics, history,
                generator.get_state(),
            )
            save_checkpoint(
                args.output_dir / "best.pt", model, optimizer, scheduler, scaler,
                epoch, config, metrics, history, generator.get_state(),
            )
            save_best_details(args.output_dir, metrics, details, epoch)
            atomic_json(args.output_dir / "best_selection_metrics.json", {
                "criterion": "maximum_validation_selection_score",
                "best_epoch": epoch, **metrics,
            })
        save_checkpoint(
            last_path, model, optimizer, scheduler, scaler, epoch, config, metrics,
            history, generator.get_state(),
        )

        resource_profile.update({
            "peak_allocated_vram_gb": (
                torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else None
            ),
            "peak_reserved_vram_gb": (
                torch.cuda.max_memory_reserved() / 1024**3 if device.type == "cuda" else None
            ),
            "train_seconds_per_epoch": float(np.mean(epoch_train_seconds)),
            "validation_seconds": float(np.mean(validation_seconds)),
            "inference_ms_per_image": inference_ms,
            "time_per_training_iteration_seconds": float(np.mean(iteration_seconds)),
        })
        atomic_json(args.output_dir / "resource_profile.json", resource_profile)
        if resource_profile["peak_reserved_vram_gb"] is not None and resource_profile["peak_reserved_vram_gb"] > 22:
            raise RuntimeError(
                f"Peak reserved VRAM {resource_profile['peak_reserved_vram_gb']:.2f} GB exceeds 22 GB gate"
            )
        completed_epochs = epoch + 1
        atomic_json(args.output_dir / "checkpoint_manifest.json", {
            "best_train": {"path": "best_train.pt", "value": best_train_loss,
                           "criterion": "minimum train_loss"},
            "best_val": {"path": "best_val.pt", "value": best_val_loss,
                         "criterion": "minimum val_loss"},
            "best_selection": {"path": "best_selection.pt", "value": best_score,
                               "criterion": "maximum validation selection_score"},
            "last": {"path": "last.pt", "epoch": epoch},
        })
        if early_stopping_bad_epochs >= args.early_stopping_patience:
            stopped_early = True
            print(
                f"EARLY_STOP: val_loss did not improve by {args.early_stopping_min_delta} "
                f"for {early_stopping_bad_epochs} epochs",
                flush=True,
            )
            break

    required_checkpoints = ("best_train.pt", "best_val.pt", "best_selection.pt", "best.pt", "last.pt")
    missing_checkpoints = [name for name in required_checkpoints if not (args.output_dir / name).is_file()]
    if missing_checkpoints:
        raise RuntimeError(f"Missing checkpoints: {missing_checkpoints}")
    if final_epoch_exclusive < args.epochs:
        atomic_json(args.output_dir / "partial_run.json", {
            "status": "INTENTIONAL_EARLY_STOP_FOR_RESUME_TEST",
            "completed_epochs": final_epoch_exclusive,
            "target_epochs": args.epochs,
            "resume_command_required": True,
        })
        return 0
    atomic_json(args.output_dir / "early_stopping.json", {
        "enabled": True, "monitor": "val_loss", "mode": "min",
        "patience": args.early_stopping_patience,
        "min_delta": args.early_stopping_min_delta,
        "stopped_early": stopped_early, "completed_epochs": completed_epochs,
        "maximum_epochs": args.epochs, "best_val_loss": best_val_loss,
    })
    atomic_json(args.output_dir / "training_completion.json", {
        "status": "completed", "run_kind": args.run_kind,
        "completed_epochs": completed_epochs, "stopped_early": stopped_early,
    })
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"FATAL: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise
