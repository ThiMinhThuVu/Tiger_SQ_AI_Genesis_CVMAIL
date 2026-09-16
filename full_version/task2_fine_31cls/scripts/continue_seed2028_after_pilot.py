#!/usr/bin/env python3
"""Gate seed 2028 folds 0-1 and conditionally submit folds 2-4.

The expansion is idempotent, validation-only, and capped at two one-GPU tasks.
It never retries failed training jobs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[3]
TASK = REPO / "full_version" / "task2_fine_31cls"
EXPERIMENT = "p3_hr640x1120_cetl_w200_seed2028"
CANDIDATE = TASK / "artifacts" / "p3_cetl_w200_seed_stability" / EXPERIMENT
REFERENCE = TASK / "artifacts" / "p3_cetl_w200_pilot" / "p3_hr640x1120_cetl_w200_seed2026"
BASELINE = TASK / "artifacts" / "p2_hr_pilot" / "p2_hr640x1120_seed2026"
DECISION_ROOT = TASK / "artifacts" / "p3_cetl_w200_seed_stability"
PILOT_DECISION = DECISION_ROOT / "seed2028_pilot_gate.json"
SUBMISSION = DECISION_ROOT / "seed2028_expansion_submission.json"
FINAL_GATE = DECISION_ROOT / "seed2028_fivefold_gate.json"
EVALUATOR = TASK / "scripts" / "evaluate_seed_stability_gate.py"
CONFIG = TASK / "configs" / "p3_hr640x1120_cetl_w200_seed2028.yaml"
MANIFEST = TASK / "splits" / "case_folds_v1.json"
OUTPUT_ROOT = TASK / "artifacts" / "p3_cetl_w200_seed_stability"
LOG_ROOT = TASK / "logs"


def run(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def atomic_json_dump(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def evaluation_command(folds: list[int], output: Path) -> list[str]:
    return [
        sys.executable,
        str(EVALUATOR),
        "--candidate-root",
        str(CANDIDATE),
        "--reference-root",
        str(REFERENCE),
        "--baseline-root",
        str(BASELINE),
        "--folds",
        *[str(fold) for fold in folds],
        "--output",
        str(output),
    ]


def main() -> None:
    run(evaluation_command([0, 1], PILOT_DECISION))
    decision = json.loads(PILOT_DECISION.read_text())
    if not decision["gates"]["all_pass"]:
        print(json.dumps({"action": "stop_after_pilot", "decision": str(PILOT_DECISION)}))
        return

    if SUBMISSION.exists():
        print(json.dumps({"action": "already_submitted", "record": str(SUBMISSION)}))
        return

    expansion_command = [
        "sbatch",
        "--parsable",
        "--job-name=tiger_p3_cetl200_s2028_f234",
        "--partition=gpu",
        "--array=2-4%2",
        "--gres=gpu:a5000:1",
        "--exclude=vishc-server-1",
        "--cpus-per-task=6",
        "--mem=55G",
        "--time=48:00:00",
        f"--chdir={REPO}",
        f"--output={LOG_ROOT}/p3_cetl200_s2028_%A_%a.log",
        f"--error={LOG_ROOT}/p3_cetl200_s2028_%A_%a.err",
        "--wrap=" + " ".join(
            [
                str(REPO / ".venv" / "bin" / "python"),
                str(TASK / "scripts" / "train_full.py"),
                "--config",
                str(CONFIG),
                "--split-manifest",
                str(MANIFEST),
                "--fold",
                "${SLURM_ARRAY_TASK_ID}",
                "--output-root",
                str(OUTPUT_ROOT),
            ]
        ),
    ]
    expansion_job_id = run(expansion_command).split(";")[0]

    final_command = evaluation_command([0, 1, 2, 3, 4], FINAL_GATE)
    final_job_id = run(
        [
            "sbatch",
            "--parsable",
            "--job-name=tiger_p3_cetl200_s2028_gate5",
            "--partition=gpu",
            f"--dependency=afterok:{expansion_job_id}",
            "--cpus-per-task=1",
            "--mem=2G",
            "--time=00:10:00",
            f"--chdir={REPO}",
            f"--output={LOG_ROOT}/p3_cetl200_s2028_gate5_%j.log",
            f"--error={LOG_ROOT}/p3_cetl200_s2028_gate5_%j.err",
            "--wrap=" + " ".join(final_command),
        ]
    ).split(";")[0]

    atomic_json_dump(
        {
            "action": "submitted_folds_2_3_4_after_pilot_pass",
            "expansion_job_id": expansion_job_id,
            "final_gate_job_id": final_job_id,
            "array": "2-4%2",
            "gpu_per_task": 1,
            "max_concurrent_gpus": 2,
            "oof_consulted": False,
        },
        SUBMISSION,
    )
    print(SUBMISSION.read_text().strip())


if __name__ == "__main__":
    main()
