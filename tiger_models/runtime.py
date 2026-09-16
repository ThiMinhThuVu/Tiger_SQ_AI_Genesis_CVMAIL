from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def git_state(root: Path) -> dict[str, Any]:
    def command(*args: str, cwd: Path = root) -> str | None:
        try:
            return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    external = {}
    for name in ("sam2", "tigersqai_evaluation"):
        checkout = root / "external" / name
        external[name] = {
            "commit": command("git", "rev-parse", "HEAD", cwd=checkout),
            "dirty": bool(command("git", "status", "--porcelain", cwd=checkout)),
        }
    return {
        "workspace_commit": command("git", "rev-parse", "HEAD"),
        "workspace_dirty": command("git", "status", "--porcelain"),
        "note": "Starting workspace contained empty .git metadata" if command("git", "rev-parse", "HEAD") is None else None,
        "external": external,
    }


def environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "hostname": platform.node(),
    }
    if torch.cuda.is_available():
        info.update({
            "gpu": torch.cuda.get_device_name(0),
            "gpu_count": torch.cuda.device_count(),
            "bf16_supported": torch.cuda.is_bf16_supported(),
        })
    return info


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    config: dict[str, Any],
    metrics: dict[str, Any],
    history: list[dict[str, Any]],
    data_generator_state: torch.Tensor | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
        "epoch": epoch, "config": config, "metrics": metrics, "history": history,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "data_generator_state": data_generator_state,
    }, temporary)
    os.replace(temporary, path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None:
        scaler.load_state_dict(payload.get("scaler", {}))
    if "torch_rng_state" in payload:
        torch.set_rng_state(payload["torch_rng_state"].cpu())
    if torch.cuda.is_available() and payload.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng_state"]])
    return payload


def update_registry(path: Path, experiment: str, **fields: Any) -> None:
    registry = yaml.safe_load(path.read_text())
    if experiment not in registry:
        raise KeyError(experiment)
    valid_status = {
        "planned", "implemented", "smoke_tested", "running", "completed",
        "failed", "not_feasible", "blocked",
    }
    if "status" in fields and fields["status"] not in valid_status:
        raise ValueError(f"Invalid registry status: {fields['status']}")
    registry[experiment].update(fields)
    descriptor, temporary = tempfile.mkstemp(prefix=".experiment_registry.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            yaml.safe_dump(registry, handle, sort_keys=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
