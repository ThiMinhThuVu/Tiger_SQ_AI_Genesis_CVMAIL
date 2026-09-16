#!/usr/bin/env python3
"""Assemble the submission image without requiring a Docker daemon.

This is a fallback for locked-down build hosts.  It appends one standards-
compliant Docker/OCI layer to an exported base image and writes an archive
that can be consumed by ``docker load``.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone


class DigestWriter(io.RawIOBase):
    def __init__(self, target):
        self.target = target
        self.digest = hashlib.sha256()
        self.position = 0

    def writable(self):
        return True

    def write(self, data):
        self.digest.update(data)
        written = self.target.write(data)
        self.position += written
        return written

    def tell(self):
        return self.position

    def flush(self):
        if not getattr(self.target, "closed", False):
            try:
                self.target.flush()
            except ValueError:
                # gzip closes its wrapped stream before RawIOBase finalization.
                pass


def json_bytes(value) -> bytes:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True).encode()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def add_bytes(archive: tarfile.TarFile, name: str, data: bytes, mtime: int):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = "root"
    info.mtime = mtime
    archive.addfile(info, io.BytesIO(data))


def root_owned(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = info.gid = 0
    info.uname = info.gname = "root"
    return info


def build_application_layer(args, layer_path: Path):
    compressed_hash = None
    uncompressed_hash = None
    with layer_path.open("wb") as raw:
        compressed = DigestWriter(raw)
        with gzip.GzipFile(fileobj=compressed, mode="wb", compresslevel=1, mtime=0) as zipped:
            uncompressed = DigestWriter(zipped)
            with tarfile.open(fileobj=uncompressed, mode="w|", format=tarfile.PAX_FORMAT) as layer:
                version_context = getattr(args, "version_context", None)
                standalone_task = getattr(args, "standalone_task", None)
                if not standalone_task:
                    for filename in ("contract.py", "validate_output.py"):
                        layer.add(
                            args.context / filename,
                            arcname=f"opt/algorithm/{filename}",
                            recursive=False,
                            filter=root_owned,
                        )

                if standalone_task:
                    layer.add(args.context / "model.py", arcname="opt/algorithm/model_base.py",
                              recursive=False, filter=root_owned)
                    layer.add(args.context / "ver1/model.py", arcname="opt/algorithm/model_ver1.py",
                              recursive=False, filter=root_owned)
                    layer.add(version_context / "model.py", arcname="opt/algorithm/model.py",
                              recursive=False, filter=root_owned)
                    task_context = version_context / standalone_task
                    for filename in ("predict.py", "validate.py"):
                        layer.add(task_context / filename, arcname=f"opt/algorithm/{filename}",
                                  recursive=False, filter=root_owned)
                elif version_context:
                    layer.add(args.context / "model.py", arcname="opt/algorithm/model_base.py",
                              recursive=False, filter=root_owned)
                    for filename in ("model.py", "predict.py"):
                        layer.add(version_context / filename, arcname=f"opt/algorithm/{filename}",
                                  recursive=False, filter=root_owned)
                else:
                    for filename in ("model.py", "predict.py"):
                        layer.add(args.context / filename, arcname=f"opt/algorithm/{filename}",
                                  recursive=False, filter=root_owned)

                if args.vendor:
                    layer.add(args.vendor, arcname="opt/algorithm/vendor", recursive=True,
                              filter=root_owned)

                for fold in range(5):
                    if standalone_task == "task1":
                        layer.add(args.task1 / f"fold_{fold}/best.pt",
                                  arcname=f"opt/checkpoints/task1/fold_{fold}/best.pt",
                                  recursive=False, filter=root_owned)
                    elif standalone_task == "task2":
                        layer.add(args.task2 / f"fold_{fold}/best.pt",
                                  arcname=f"opt/checkpoints/task2/fold_{fold}/best.pt",
                                  recursive=False, filter=root_owned)
                    elif standalone_task == "task3":
                        layer.add(args.task2 / f"fold_{fold}/best.pt",
                                  arcname=f"opt/checkpoints/encoder/fold_{fold}/best.pt",
                                  recursive=False, filter=root_owned)
                        layer.add(args.task3 / f"fold_{fold}/best_visibility.pt",
                                  arcname=f"opt/checkpoints/head/fold_{fold}/best_visibility.pt",
                                  recursive=False, filter=root_owned)
                    elif args.task1:
                        layer.add(args.task1 / f"fold_{fold}/best.pt",
                                  arcname=f"opt/checkpoints/task1/fold_{fold}/best.pt",
                                  recursive=False, filter=root_owned)
                        layer.add(args.task2 / f"fold_{fold}/best.pt",
                                  arcname=f"opt/checkpoints/task2/fold_{fold}/best.pt",
                                  recursive=False, filter=root_owned)
                        layer.add(args.task3 / f"fold_{fold}/best_visibility.pt",
                                  arcname=f"opt/checkpoints/task3/fold_{fold}/best_visibility.pt",
                                  recursive=False, filter=root_owned)
                    else:
                        layer.add(args.segmentation / f"fold_{fold}/best.pt",
                                  arcname=f"opt/checkpoints/segmentation/fold_{fold}/best.pt",
                                  recursive=False, filter=root_owned)
                        layer.add(args.visibility / f"fold_{fold}/best_visibility.pt",
                                  arcname=f"opt/checkpoints/visibility/fold_{fold}/best_visibility.pt",
                                  recursive=False, filter=root_owned)
            uncompressed_hash = uncompressed.digest.hexdigest()
        compressed_hash = compressed.digest.hexdigest()
    return compressed_hash, uncompressed_hash, layer_path.stat().st_size


def assemble(args):
    created = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    mtime = int(datetime.now(timezone.utc).timestamp())

    with tarfile.open(args.base, "r:") as base:
        index = json.load(base.extractfile("index.json"))
        base_manifest_digest = index["manifests"][0]["digest"].split(":", 1)[1]
        base_manifest_path = f"blobs/sha256/{base_manifest_digest}"
        base_manifest = json.load(base.extractfile(base_manifest_path))
        base_config_digest = base_manifest["config"]["digest"].split(":", 1)[1]
        base_config_path = f"blobs/sha256/{base_config_digest}"
        config = json.load(base.extractfile(base_config_path))

        with tempfile.TemporaryDirectory(prefix="tigersqai-layer-") as temporary:
            layer_path = Path(temporary) / "application.tar.gz"
            layer_digest, diff_id, layer_size = build_application_layer(args, layer_path)

            config = copy.deepcopy(config)
            config["created"] = created
            runtime = config.setdefault("config", {})
            env_updates = {
                "PYTHONUNBUFFERED": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "HF_HOME": "/opt/huggingface",
                "TORCH_HOME": "/opt/checkpoints",
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONPATH": "/opt/algorithm/vendor",
            }
            if args.task2_height is not None:
                env_updates["TIGERSQAI_TASK2_HEIGHT"] = str(args.task2_height)
                env_updates["TIGERSQAI_TASK2_WIDTH"] = str(args.task2_width)
            if args.task3_height is not None:
                env_updates["TIGERSQAI_TASK3_HEIGHT"] = str(args.task3_height)
                env_updates["TIGERSQAI_TASK3_WIDTH"] = str(args.task3_width)
            old_env = runtime.get("Env", [])
            retained = [item for item in old_env if item.split("=", 1)[0] not in env_updates]
            runtime["Env"] = retained + [f"{key}={value}" for key, value in env_updates.items()]
            runtime["WorkingDir"] = "/opt/algorithm"
            runtime["Entrypoint"] = ["python", "/opt/algorithm/predict.py"]
            runtime.pop("Cmd", None)
            config["rootfs"]["diff_ids"].append(f"sha256:{diff_id}")
            config.setdefault("history", []).append(
                {
                    "created": created,
                    "created_by": "TIGER SQ-AI submission application, dependencies, and checkpoints",
                    "comment": "daemonless deterministic assembly",
                }
            )
            config_data = json_bytes(config)
            config_digest = sha256_bytes(config_data)

            manifest = copy.deepcopy(base_manifest)
            manifest["config"] = {
                "mediaType": "application/vnd.docker.container.image.v1+json",
                "digest": f"sha256:{config_digest}",
                "size": len(config_data),
            }
            manifest["layers"].append(
                {
                    "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
                    "digest": f"sha256:{layer_digest}",
                    "size": layer_size,
                }
            )
            manifest_data = json_bytes(manifest)
            manifest_digest = sha256_bytes(manifest_data)

            index = {
                "schemaVersion": 2,
                "mediaType": "application/vnd.oci.image.index.v1+json",
                "manifests": [
                    {
                        "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
                        "digest": f"sha256:{manifest_digest}",
                        "size": len(manifest_data),
                        "annotations": {
                            "io.containerd.image.name": args.tag,
                            "org.opencontainers.image.created": created,
                            "org.opencontainers.image.ref.name": args.tag.rsplit(":", 1)[-1],
                        },
                        "platform": {"architecture": "amd64", "os": "linux"},
                    }
                ],
            }
            index_data = json_bytes(index)
            docker_manifest = json_bytes(
                [
                    {
                        "Config": f"blobs/sha256/{config_digest}",
                        "RepoTags": [args.tag],
                        "Layers": [
                            *(f"blobs/sha256/{entry['digest'].split(':', 1)[1]}" for entry in base_manifest["layers"]),
                            f"blobs/sha256/{layer_digest}",
                        ],
                    }
                ]
            )

            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary_output = args.output.with_suffix(args.output.suffix + ".partial")
            with tarfile.open(temporary_output, "w", format=tarfile.PAX_FORMAT) as output:
                for member in base:
                    if member.name in {"index.json", "manifest.json", "oci-layout"}:
                        continue
                    stream = base.extractfile(member) if member.isfile() else None
                    output.addfile(member, stream)

                with layer_path.open("rb") as stream:
                    info = tarfile.TarInfo(f"blobs/sha256/{layer_digest}")
                    info.size = layer_size
                    info.mode = 0o644
                    info.uid = info.gid = 0
                    info.mtime = mtime
                    output.addfile(info, stream)

                add_bytes(output, f"blobs/sha256/{config_digest}", config_data, mtime)
                add_bytes(output, f"blobs/sha256/{manifest_digest}", manifest_data, mtime)
                add_bytes(output, "index.json", index_data, mtime)
                add_bytes(output, "manifest.json", docker_manifest, mtime)
                add_bytes(output, "oci-layout", b'{"imageLayoutVersion":"1.0.0"}', mtime)

            os.replace(temporary_output, args.output)

    print(json.dumps({
        "archive": str(args.output),
        "tag": args.tag,
        "layer_digest": f"sha256:{layer_digest}",
        "diff_id": f"sha256:{diff_id}",
        "manifest_digest": f"sha256:{manifest_digest}",
    }, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--vendor", type=Path)
    parser.add_argument("--segmentation", type=Path)
    parser.add_argument("--visibility", type=Path)
    parser.add_argument("--version-context", type=Path)
    parser.add_argument("--task1", type=Path)
    parser.add_argument("--task2", type=Path)
    parser.add_argument("--task3", type=Path)
    parser.add_argument("--standalone-task", choices=("task1", "task2", "task3"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag", default="tigersqai26_submission:v1")
    parser.add_argument("--task2-height", type=int)
    parser.add_argument("--task2-width", type=int)
    parser.add_argument("--task3-height", type=int)
    parser.add_argument("--task3-width", type=int)
    args = parser.parse_args()
    if (args.task2_height is None) != (args.task2_width is None):
        parser.error("provide both --task2-height and --task2-width")
    if (args.task3_height is None) != (args.task3_width is None):
        parser.error("provide both --task3-height and --task3-width")
    legacy = bool(args.segmentation and args.visibility)
    ver1 = bool(args.version_context and args.task1 and args.task2 and args.task3)
    standalone = bool(
        args.version_context
        and args.standalone_task
        and (
            (args.standalone_task == "task1" and args.task1)
            or (args.standalone_task == "task2" and args.task2)
            or (args.standalone_task == "task3" and args.task2 and args.task3)
        )
    )
    if sum((legacy, ver1, standalone)) != 1:
        parser.error("provide exactly one legacy, combined-ver1, or standalone-ver2 input set")
    return args


if __name__ == "__main__":
    assemble(parse_args())
