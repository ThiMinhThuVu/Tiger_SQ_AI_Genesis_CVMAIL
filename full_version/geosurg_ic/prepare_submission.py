#!/usr/bin/env python3
"""Build and smoke-test the two authorized five-fold Task-2 submissions."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from full_version.geosurg_ic.run_pilot import save_json
from full_version.geosurg_ic.evaluate_native import aggregate

PILOT = ROOT / "full_version/geosurg_ic/artifacts/pilot_v1"
VERSIONS = {"aug": "ver6_task2_aug", "geo": "ver7_task2_geosurg_ic"}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def five_fold_summary(arm):
    rows, checkpoints = [], []
    for fold in range(5):
        path = PILOT / arm / f"fold_{fold}"
        completion = json.loads((path / "completion.json").read_text())
        provenance = json.loads((path / "provenance.json").read_text())
        assert completion["status"] == "completed" and completion["steps"] == 192
        assert provenance["gpu_count"] == 1
        assert not set(provenance["train_cases"]) & set(provenance["validation_cases"])
        metric = json.loads((path / "validation.json").read_text())["metrics"]["fine"]
        assert set(metric["per_case_dice"]) == set(provenance["validation_cases"])
        for case, d in metric["per_case_dice"].items():
            n = metric["per_case_nhd"][case]
            rows.append({"case": case, "dice": d, "nhd": n, "score": (d+1-n)/2})
        checkpoints.append({"fold": fold, "path": str(path / "best.pt"),
                            "sha256": digest(path / "best.pt")})
    assert len(rows) == 40 and len({r["case"] for r in rows}) == 40
    assert len({c["sha256"] for c in checkpoints}) == 5
    per_center = {}
    for center in sorted({r["case"].split("_case_")[0] for r in rows}):
        subset = [r for r in rows if r["case"].split("_case_")[0] == center]
        per_center[center] = {k: float(np.mean([r[k] for r in subset])) for k in ("dice", "nhd", "score")}
    return {"arm": arm, "folds": 5, "cases": 40, "grid": [640, 1120],
            "split": "OOF validation; original checkpoints selected using these validation folds",
            "case_macro": {k: float(np.mean([r[k] for r in rows])) for k in ("dice", "nhd", "score")},
            "center_macro": {k: float(np.mean([r[k] for r in per_center.values()])) for k in ("dice", "nhd", "score")},
            "per_case": {r["case"]: {k: v for k, v in r.items() if k != "case"} for r in rows},
            "per_center": per_center, "checkpoints": checkpoints}


def extract_verified_application(archive, runtime, checkpoints):
    """Read only the final layer and extract explicitly allowlisted files."""
    runtime.mkdir(parents=True, exist_ok=False)
    expected = {f"opt/checkpoints/task2/fold_{r['fold']}/best.pt": r["sha256"] for r in checkpoints}
    expected_code = {
        "opt/algorithm/model_base.py": ROOT / "submission/model.py",
        "opt/algorithm/model_ver1.py": ROOT / "submission/ver1/model.py",
        "opt/algorithm/model.py": ROOT / "submission/ver2/model.py",
        "opt/algorithm/predict.py": ROOT / "submission/ver2/task2/predict.py",
        "opt/algorithm/validate.py": ROOT / "submission/ver2/task2/validate.py",
    }
    seen = set()
    with tarfile.open(archive, "r:") as outer:
        index = json.load(outer.extractfile("index.json"))
        manifest_entry = index["manifests"][0]
        assert manifest_entry["platform"] == {"architecture": "amd64", "os": "linux"}
        manifest_data = outer.extractfile("blobs/sha256/" + manifest_entry["digest"].split(":")[1]).read()
        assert hashlib.sha256(manifest_data).hexdigest() == manifest_entry["digest"].split(":")[1]
        manifest = json.loads(manifest_data)
        config_data = outer.extractfile("blobs/sha256/" + manifest["config"]["digest"].split(":")[1]).read()
        assert hashlib.sha256(config_data).hexdigest() == manifest["config"]["digest"].split(":")[1]
        config = json.loads(config_data)
        assert config["config"]["Entrypoint"] == ["python", "/opt/algorithm/predict.py"]
        env = dict(v.split("=", 1) for v in config["config"]["Env"])
        assert env["HF_HUB_OFFLINE"] == "1"
        assert env["TIGERSQAI_TASK2_HEIGHT"] == "640" and env["TIGERSQAI_TASK2_WIDTH"] == "1120"
        layer = manifest["layers"][-1]
        layer_name = "blobs/sha256/" + layer["digest"].split(":")[1]
        layer_sha = hashlib.sha256()
        with outer.extractfile(layer_name) as source:
            for block in iter(lambda: source.read(8 << 20), b""):
                layer_sha.update(block)
        assert layer_sha.hexdigest() == layer["digest"].split(":")[1]
        with tarfile.open(fileobj=outer.extractfile(layer_name), mode="r|gz") as contents:
            for member in contents:
                if member.name not in expected and member.name not in expected_code:
                    raise ValueError(f"Unexpected application-layer file: {member.name}")
                assert member.isfile() and member.name not in seen
                seen.add(member.name)
                destination = runtime / member.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                with contents.extractfile(member) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target, 8 << 20)
                wanted = expected.get(member.name) or digest(expected_code[member.name])
                assert digest(destination) == wanted
        assert seen == set(expected) | set(expected_code)
    return {"manifest_sha256": manifest_entry["digest"], "application_layer_sha256": layer["digest"],
            "verified_files": sorted(seen), "architecture": "linux/amd64"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=ROOT / "full_version/geosurg_ic/submission")
    args = p.parse_args()
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = {arm: five_fold_summary(arm) for arm in VERSIONS}
    # Only the final submission per task is evaluated by the challenge.
    order = sorted(VERSIONS, key=lambda arm: summaries[arm]["center_macro"]["score"])
    save_json(args.output / "five_fold_summary.json", summaries)
    print(json.dumps({"five_fold_scores": {a: s["center_macro"] for a, s in summaries.items()},
                      "submission_order": order, "final_candidate": order[-1]}), flush=True)
    inputs = args.output / "smoke_input"
    inputs.mkdir(exist_ok=True)
    sizes = set()
    for source in sorted((ROOT / "data/images").glob("*.png")):
        with Image.open(source) as im:
            size = im.size
        if size not in sizes:
            sizes.add(size)
            destination = inputs / source.name
            if not destination.exists():
                shutil.copy2(source, destination)
        if len(sizes) >= 3:
            break
    candidates = []
    for arm in order:
        version = VERSIONS[arm]
        archive = ROOT / "submission/dist" / f"tigersqai_Genesis_CVMAIL_{version}.tar"
        if archive.exists() or archive.with_suffix(".tar.gz").exists():
            raise FileExistsError(archive)
        started = time.monotonic()
        subprocess.run([sys.executable, str(ROOT / "submission/assemble_image.py"),
            "--base", str(ROOT / "submission/dist/tigersqai_submission_task123_v1.tar"),
            "--context", str(ROOT / "submission"), "--version-context", str(ROOT / "submission/ver2"),
            "--standalone-task", "task2", "--task2", str(PILOT / arm),
            "--task2-height", "640", "--task2-width", "1120", "--output", str(archive),
            "--tag", "tigersqai26_genesis_cvmail:" + version.replace("_", "-")], check=True)
        runtime = args.output / (arm + "_runtime")
        audit = extract_verified_application(archive, runtime, summaries[arm]["checkpoints"])
        env = os.environ.copy()
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                   PYTHONDONTWRITEBYTECODE="1", TIGERSQAI_INPUT_DIR=str(inputs),
                   TIGERSQAI_OUTPUT_DIR=str(runtime / "output"),
                   TIGERSQAI_CHECKPOINT_DIR=str(runtime / "opt/checkpoints/task2"),
                   TIGERSQAI_TASK2_HEIGHT="640", TIGERSQAI_TASK2_WIDTH="1120",
                   PYTHONPATH=str(runtime / "opt/algorithm"))
        smoke_start = time.monotonic()
        subprocess.run([sys.executable, str(runtime / "opt/algorithm/predict.py")],
                       cwd=runtime, env=env, check=True, timeout=600)
        audit.update(smoke_passed=True, smoke_frames=len(sizes),
                     smoke_seconds=time.monotonic()-smoke_start,
                     smoke_mode="extracted application layer and exact checkpoints, host pinned-dependency venv; Docker socket unavailable")
        compressed = archive.with_suffix(".tar.gz")
        with archive.open("rb") as source, compressed.open("xb") as target:
            with gzip.GzipFile(fileobj=target, mode="wb", compresslevel=1, mtime=0) as gz:
                shutil.copyfileobj(source, gz, 8 << 20)
        sha = digest(compressed)
        (compressed.with_suffix(compressed.suffix + ".sha256")).write_text(sha + "  " + compressed.name + "\n")
        candidate = {"arm": arm, "version": version, "archive": str(compressed), "sha256": sha,
                     "bytes": compressed.stat().st_size, "evaluation_id": "9619534", "team_id": "3602194",
                     "parent_id": "syn77263888", "summary": summaries[arm], "audit": audit,
                     "preparation_seconds": time.monotonic()-started}
        candidates.append(candidate)
        save_json(args.output / (arm + "_prepared.json"), candidate)
        print(json.dumps({"prepared": version, "sha256": sha, "seconds": time.monotonic()-started}), flush=True)
    save_json(args.output / "upload_plan.json", {"candidates": candidates,
        "final_candidate": order[-1], "challenge_policy": "Only the final submission per team and task is evaluated",
        "authorization": "User explicitly requested both candidates trained over 5 folds and submitted to challenge"})


if __name__ == "__main__":
    main()
