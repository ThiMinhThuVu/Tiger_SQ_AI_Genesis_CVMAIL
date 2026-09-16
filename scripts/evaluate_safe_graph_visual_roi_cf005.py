#!/usr/bin/env python3
"""Evaluate whether frozen visual evidence adds information beyond P2-DENSE."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))
from scripts.analyze_safe_graph_p2_cf005 import (  # noqa: E402
    bootstrap_deltas,
    macro_metrics,
    parse_bool,
    summarize_operational,
)
from scripts.evaluate_safe_graph_pair_specific import threshold_metrics  # noqa: E402
from scripts.train_safe_graph_warning_baselines import binary_metrics  # noqa: E402


METHODS = {
    "P2_DENSE_RAW": {"kind": "raw", "scalar": ["dense_counterpart_probability"]},
    "V0_GEOMETRY_DIRECTION": {
        "kind": "learned",
        "scalar": [
            "pair_class_indicator", "roi_area_fraction", "roi_bbox_aspect",
            "roi_bbox_fill", "roi_centroid_x", "roi_centroid_y", "roi_compactness",
            "roi_low_token_support",
        ],
    },
    "V1_ROI_ONLY": {"kind": "learned", "embedding": ["roi"]},
    "V2_P2_LEARNED": {
        "kind": "learned", "scalar": ["dense_counterpart_probability"],
    },
    "V3_P2_GEOMETRY": {
        "kind": "learned",
        "scalar": [
            "dense_counterpart_probability", "pair_class_indicator",
            "roi_area_fraction", "roi_bbox_aspect", "roi_bbox_fill",
            "roi_centroid_x", "roi_centroid_y", "roi_compactness",
            "roi_low_token_support",
        ],
    },
    "V4_P2_ROI": {
        "kind": "learned", "scalar": ["dense_counterpart_probability"],
        "embedding": ["roi"],
    },
    "V5_P2_GLOBAL": {
        "kind": "learned", "scalar": ["dense_counterpart_probability"],
        "embedding": ["global_cls"],
    },
    "V6_P2_SHUFFLED_ROI": {
        "kind": "learned", "scalar": ["dense_counterpart_probability"],
        "embedding": ["shuffled_roi"],
    },
    "V7_P2_ROI_CONTRAST": {
        "kind": "learned", "scalar": ["dense_counterpart_probability"],
        "embedding": ["roi", "contrast"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--visual-root", type=Path,
        default=ROOT / "artifacts/safe_graph_verifier/visual_roi_cf005_v1",
    )
    parser.add_argument(
        "--pair-cohort", type=Path,
        default=ROOT / "artifacts/safe_graph_verifier/p2_dense_cf005_v1/dense_pair_cohort.csv",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pca-components", type=int, default=8)
    parser.add_argument("--bootstrap-replicates", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=2026)
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


def component_key(row: dict) -> tuple[str, str, str, str]:
    return str(row["fold"]), row["case_id"], row["frame"], str(row["component_id"])


def numeric_value(value) -> float:
    if isinstance(value, bool) or str(value).strip().lower() in {"true", "false"}:
        return float(parse_bool(value))
    return float(value)


def load_inputs(visual_root: Path, pair_path: Path):
    manifest = json.loads((visual_root / "manifest.json").read_text())
    if manifest.get("gt_used_as_feature") is not False or manifest.get("test_data_used") is not False:
        raise ValueError("Visual artifact violates the no-GT/no-test contract")
    if manifest.get("encoder_frozen") is not True:
        raise ValueError("Visual encoder must be frozen")

    index = read_csv(visual_root / "component_index.csv")
    pair = read_csv(pair_path)
    if len(index) != 744 or len(pair) != 744:
        raise ValueError("Expected frozen CF005 cohort with 744 components")
    pair_lookup = {component_key(row): row for row in pair}
    if len(pair_lookup) != len(pair):
        raise ValueError("Duplicate component keys in pair cohort")
    rows = []
    for row in index:
        source = pair_lookup.get(component_key(row))
        if source is None:
            raise ValueError(f"Visual component not found in pair cohort: {component_key(row)}")
        if row["pair_swap_target"] != source["pair_swap_target"]:
            raise ValueError("Target mismatch between visual index and pair cohort")
        rows.append({**source, **row, "pair_class_indicator": int(row["predicted_class_id"]) == 19})
    if len({component_key(row) for row in rows}) != 744:
        raise ValueError("Visual index has duplicate or missing component keys")

    with np.load(visual_root / "component_embeddings.npz") as arrays:
        embeddings = {name: arrays[name].astype(np.float32) for name in arrays.files}
    expected = {"roi", "ring", "contrast", "global_cls"}
    if set(embeddings) != expected:
        raise ValueError(f"Unexpected embedding arrays: {sorted(embeddings)}")
    if any(value.shape != (744, 384) for value in embeddings.values()):
        raise ValueError("Expected four 744x384 embedding matrices")
    return rows, embeddings, manifest


def make_shuffled_roi(rows: list[dict], roi: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Shuffle within case and predicted class, preserving nuisance distributions."""
    rng = np.random.default_rng(seed)
    permutation = np.arange(len(rows))
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[(row["case_id"], row["predicted_class_id"])].append(index)
    for indices in groups.values():
        indices = np.asarray(indices)
        permutation[indices] = rng.permutation(indices)
    return roi[permutation], permutation


def design_matrices(
    rows: list[dict], embeddings: dict[str, np.ndarray], specification: dict,
    train: np.ndarray, test: np.ndarray, pca_components: int,
):
    train_blocks = []
    test_blocks = []
    audit = {"pca_explained_variance": None, "features_after_transform": 0}
    scalar_names = specification.get("scalar", [])
    if scalar_names:
        scalar = np.asarray([[numeric_value(row[name]) for name in scalar_names] for row in rows])
        scaler = StandardScaler().fit(scalar[train])
        train_blocks.append(scaler.transform(scalar[train]))
        test_blocks.append(scaler.transform(scalar[test]))
    embedding_names = specification.get("embedding", [])
    if embedding_names:
        matrix = np.concatenate([embeddings[name] for name in embedding_names], axis=1)
        scaler = StandardScaler().fit(matrix[train])
        train_embedding = scaler.transform(matrix[train])
        test_embedding = scaler.transform(matrix[test])
        components = min(pca_components, train_embedding.shape[0] - 1, train_embedding.shape[1])
        pca = PCA(n_components=components, random_state=0).fit(train_embedding)
        train_blocks.append(pca.transform(train_embedding))
        test_blocks.append(pca.transform(test_embedding))
        audit["pca_explained_variance"] = float(pca.explained_variance_ratio_.sum())
    x_train = np.concatenate(train_blocks, axis=1)
    x_test = np.concatenate(test_blocks, axis=1)
    audit["features_after_transform"] = x_train.shape[1]
    return x_train, x_test, audit


def evaluate(rows, embeddings, pca_components: int, coverages: list[float], seed: int):
    case_ids = np.asarray([row["case_id"] for row in rows])
    cases = sorted(set(case_ids))
    y = np.asarray([parse_bool(row["pair_swap_target"]) for row in rows], dtype=np.uint8)
    risks = {}
    thresholds = {}
    audits = []
    for method, specification in METHODS.items():
        risk = np.zeros(len(rows), dtype=float)
        for held_case in cases:
            train = case_ids != held_case
            test = ~train
            if specification["kind"] == "raw":
                values = np.asarray([float(row[specification["scalar"][0]]) for row in rows])
                train_risk = values[train]
                risk[test] = values[test]
                audits.append({
                    "method": method, "held_case": held_case, "training_components": int(train.sum()),
                    "test_components": int(test.sum()), "pca_explained_variance": None,
                    "features_after_transform": 1,
                })
            else:
                x_train, x_test, audit = design_matrices(
                    rows, embeddings, specification, train, test, pca_components,
                )
                model = LogisticRegression(
                    max_iter=3000, class_weight="balanced", random_state=seed, solver="liblinear",
                ).fit(x_train, y[train])
                train_risk = model.predict_proba(x_train)[:, 1]
                risk[test] = model.predict_proba(x_test)[:, 1]
                audits.append({
                    "method": method, "held_case": held_case, "training_components": int(train.sum()),
                    "test_components": int(test.sum()), **audit,
                })
            for coverage in coverages:
                thresholds[(method, held_case, coverage)] = float(
                    np.quantile(train_risk, 1.0 - coverage)
                )
        risks[method] = risk

    metrics = macro_metrics(y, risks, case_ids)
    operational = []
    predictions = []
    for method, risk in risks.items():
        for held_case in cases:
            selected = case_ids == held_case
            for coverage in coverages:
                operational.append({
                    "method": method, "target_coverage": coverage, "case_id": held_case,
                    **threshold_metrics(y[selected], risk[selected], thresholds[(method, held_case, coverage)]),
                })
        for index, row in enumerate(rows):
            predictions.append({
                "method": method, "fold": row["fold"], "case_id": row["case_id"],
                "frame": row["frame"], "component_id": row["component_id"],
                "predicted_class_id": row["predicted_class_id"], "area_bin": row["area_bin"],
                "roi_low_token_support": row["roi_low_token_support"],
                "roi_mean_tool_probability": row["roi_mean_tool_probability"],
                "true_label": int(y[index]), "risk": float(risk[index]),
            })
    return y, case_ids, risks, metrics, operational, predictions, audits


def subgroup_metrics(rows, y, risks, case_ids):
    groups = {
        "all": np.ones(len(rows), dtype=bool),
        "direction_pred_7": np.asarray([int(row["predicted_class_id"]) == 7 for row in rows]),
        "direction_pred_19": np.asarray([int(row["predicted_class_id"]) == 19 for row in rows]),
        "area_tiny": np.asarray([row["area_bin"] == "tiny" for row in rows]),
        "area_non_tiny": np.asarray([row["area_bin"] != "tiny" for row in rows]),
        "token_support_low": np.asarray([parse_bool(row["roi_low_token_support"]) for row in rows]),
        "token_support_adequate": np.asarray([not parse_bool(row["roi_low_token_support"]) for row in rows]),
        "tool_low": np.asarray([float(row["roi_mean_tool_probability"]) < 0.10 for row in rows]),
        "tool_high": np.asarray([float(row["roi_mean_tool_probability"]) >= 0.10 for row in rows]),
    }
    output = []
    for group, selected in groups.items():
        for method, risk in risks.items():
            pooled = binary_metrics(y[selected], risk[selected])
            case_values = []
            for case_id in sorted(set(case_ids[selected])):
                subset = selected & (case_ids == case_id)
                result = binary_metrics(y[subset], risk[subset])
                if result["auprc"] is not None:
                    case_values.append(result["auprc"])
            output.append({
                "group": group, "method": method, **pooled,
                "evaluable_cases": len(case_values),
                "macro_case_auprc": float(np.mean(case_values)) if case_values else None,
            })
    return output


def gate_decision(metrics, subgroups, bootstrap, operational_summary):
    macro = {row["method"]: row for row in metrics if row["scope"] == "macro_case_mean"}
    boot = {row["method"]: row for row in bootstrap}
    subgroup = {(row["group"], row["method"]): row for row in subgroups}
    operational = {
        (row["method"], float(row["target_coverage"])): row for row in operational_summary
    }
    primary = "V4_P2_ROI"
    reference = "V2_P2_LEARNED"
    delta = float(macro[primary]["auprc"]) - float(macro[reference]["auprc"])
    non_tiny_delta = (
        float(subgroup[("area_non_tiny", primary)]["macro_case_auprc"])
        - float(subgroup[("area_non_tiny", reference)]["macro_case_auprc"])
    )
    ppv_deltas = {}
    for coverage in [0.05, 0.10]:
        current = operational[(primary, coverage)]["min_case_ppv"]
        baseline = operational[(reference, coverage)]["min_case_ppv"]
        ppv_deltas[str(coverage)] = (
            float(current) - float(baseline) if current is not None and baseline is not None else None
        )
    primary_core = (
        delta >= 0.02 and int(boot[primary]["cases_improved"]) >= 4
        and float(boot[primary]["case_bootstrap_ci_low"]) > 0
    )
    reproduced = []
    for control in ["V3_P2_GEOMETRY", "V5_P2_GLOBAL", "V6_P2_SHUFFLED_ROI"]:
        control_delta = float(macro[control]["auprc"]) - float(macro[reference]["auprc"])
        if (
            control_delta >= 0.02 and int(boot[control]["cases_improved"]) >= 4
            and float(boot[control]["case_bootstrap_ci_low"]) > 0
        ):
            reproduced.append(control)
    ppv_safe = all(value is not None and value >= -0.05 for value in ppv_deltas.values())
    passed = primary_core and non_tiny_delta >= -0.01 and not reproduced and ppv_safe
    return {
        "status": "VISUAL_INFORMATION_GAIN_GO" if passed else "VISUAL_INFORMATION_GAIN_NO_GO",
        "primary": primary,
        "reference": reference,
        "macro_case_delta_auprc": delta,
        "cases_improved": int(boot[primary]["cases_improved"]),
        "case_bootstrap_ci_low": float(boot[primary]["case_bootstrap_ci_low"]),
        "case_bootstrap_ci_high": float(boot[primary]["case_bootstrap_ci_high"]),
        "non_tiny_macro_case_delta_auprc": non_tiny_delta,
        "control_methods_reproducing_gate": reproduced,
        "min_case_ppv_deltas": ppv_deltas,
        "material_ppv_degradation_definition": "absolute min-case PPV delta below -0.05",
        "passed": passed,
    }


def render_readme(manifest: dict, macro: dict[str, dict], gate: dict) -> str:
    rows = []
    for method in METHODS:
        delta = float(macro[method]["auprc"]) - float(macro["V2_P2_LEARNED"]["auprc"])
        rows.append(
            f"| {method} | {float(macro[method]['auprc']):.6f} | {delta:+.6f} | "
            f"{float(macro[method]['auroc']):.6f} |"
        )
    return "\n".join([
        "# CF005 visual ROI information-gain evaluation", "",
        "## Material Passport", "",
        "- Origin Skill: academic-research-suite / experiment-agent",
        "- Verification Status: exploratory, OOF validation only",
        f"- Decision: **{gate['status']}**", "", "## Macro-case comparison", "",
        "| Method | AUPRC | Delta vs V2 | AUROC |", "|---|---:|---:|---:|", *rows, "",
        "## Frozen primary decision", "",
        f"- V4 − V2 AUPRC: `{gate['macro_case_delta_auprc']:+.6f}`.",
        f"- Cases improved: `{gate['cases_improved']}/5`.",
        f"- Descriptive case-bootstrap interval: "
        f"`[{gate['case_bootstrap_ci_low']:.6f}, {gate['case_bootstrap_ci_high']:.6f}]`.",
        f"- Non-tiny delta: `{gate['non_tiny_macro_case_delta_auprc']:+.6f}`.",
        f"- Controls reproducing the gate: `{gate['control_methods_reproducing_gate']}`.", "",
        "This is an exploratory component-warning experiment. It does not authorize clinical "
        "deployment, automatic correction, or ontology-loss training.", "",
    ])


def main() -> None:
    args = parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output root: {args.output_root}")
    rows, embeddings, visual_manifest = load_inputs(args.visual_root, args.pair_cohort)
    shuffled, permutation = make_shuffled_roi(rows, embeddings["roi"], args.seed)
    embeddings["shuffled_roi"] = shuffled
    coverages = [0.05, 0.10, 0.20, 0.30]
    y, case_ids, risks, metrics, operational, predictions, audits = evaluate(
        rows, embeddings, args.pca_components, coverages, args.seed,
    )
    operational_summary = summarize_operational(operational)
    subgroups = subgroup_metrics(rows, y, risks, case_ids)
    bootstrap = bootstrap_deltas(
        metrics, [method for method in METHODS if method != "V2_P2_LEARNED"],
        "V2_P2_LEARNED", args.bootstrap_replicates, args.seed,
    )
    gate = gate_decision(metrics, subgroups, bootstrap, operational_summary)
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "metrics.csv", metrics)
    write_csv(args.output_root / "operational_metrics.csv", operational)
    write_csv(args.output_root / "operational_summary.csv", operational_summary)
    write_csv(args.output_root / "subgroup_metrics.csv", subgroups)
    write_csv(args.output_root / "case_bootstrap_deltas.csv", bootstrap)
    write_csv(args.output_root / "oof_risk_predictions.csv", predictions)
    write_csv(args.output_root / "fold_transform_audit.csv", audits)
    write_csv(args.output_root / "shuffled_roi_permutation.csv", [
        {"row_index": index, "source_row_index": int(source)}
        for index, source in enumerate(permutation)
    ])
    macro = {row["method"]: row for row in metrics if row["scope"] == "macro_case_mean"}
    manifest = {
        "artifact": "CF005 frozen visual ROI information-gain evaluation",
        "version": args.output_root.name,
        **gate,
        "source_visual_artifact": str(args.visual_root),
        "source_visual_manifest": visual_manifest,
        "methods": METHODS,
        "outer_evaluation": "leave-one-case-out across five OOF validation cases",
        "preprocessing": f"train-only StandardScaler + PCA({args.pca_components})",
        "classifier": "class-balanced L2 logistic regression, frozen settings",
        "shuffled_control": "within case and predicted class; deterministic; no labels",
        "test_data_used": False,
        "selection_leakage": "CF005 selected from complete OOF aggregate; exploratory only",
        "clinical_warning_ready": False,
        "correction_authorized": False,
        "ontology_loss_authorized": False,
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output_root / "README.md").write_text(render_readme(manifest, macro, gate))
    print(json.dumps({
        "status": gate["status"],
        "p2_dense_raw_macro_case_auprc": float(macro["P2_DENSE_RAW"]["auprc"]),
        "v2_matched_macro_case_auprc": float(macro["V2_P2_LEARNED"]["auprc"]),
        "v4_primary_macro_case_auprc": float(macro["V4_P2_ROI"]["auprc"]),
        **{key: value for key, value in gate.items() if key not in {"status", "primary", "reference"}},
    }, indent=2))


if __name__ == "__main__":
    main()
