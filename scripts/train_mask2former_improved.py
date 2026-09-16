#!/usr/bin/env python3
"""Stage-wise training for the TIGER Mask2Former improvement experiments."""
from __future__ import annotations

import argparse, json, random, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_mask2former import build_model, collate
from tiger_models.data import (STATIONS, LabelMap, TigerMultitaskDataset,
    records_for_fold, visibility_pos_weight)
from tiger_models.cetl_loss import cross_entropy_tversky_loss
from tiger_models.improved_mask2former import (ImprovedMask2Former, aggregate_fine_to_coarse,
    presence_pos_weight, presence_targets, soft_presence_modulation, symmetric_kl)
from tiger_models.metrics import (segmentation_failure_metrics, segmentation_metrics,
    visibility_metrics)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--run-all-folds", action="store_true"); parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true"); parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts/mask2former_improved")
    return parser.parse_args()


def station_anatomy_prior(records, labelmap, smoothing=1.0):
    """Estimate P(anatomy present | station visible) from one training split only."""
    station_counts = np.zeros(len(STATIONS), dtype=np.float64)
    cooccurrence = np.zeros((len(STATIONS), 31), dtype=np.float64)
    for record in records:
        with Image.open(record.fine_mask) as handle:
            fine = labelmap.decode_fine(np.asarray(handle.convert("RGB")))
        anatomy = (np.bincount(fine.ravel(), minlength=31) > 0).astype(np.float64)
        visible = record.visibility.astype(np.float64)
        station_counts += visible
        cooccurrence += visible[:, None] * anatomy[None, :]
    alpha = float(smoothing)
    if alpha < 0:
        raise ValueError("context_prior_smoothing must be non-negative")
    prior = (cooccurrence + alpha) / (station_counts[:, None] + 2.0 * alpha).clip(min=1.0)
    return torch.from_numpy(prior.astype(np.float32))


def load_sink_margin_matrix(config, num_labels):
    path = config.get("sink_margin_graph_path")
    if not path:
        return None
    payload = json.loads(Path(path).read_text())
    matrix = np.asarray(payload["matrix"], dtype=np.float32)
    if matrix.shape != (num_labels, num_labels):
        raise ValueError(f"sink_margin_graph_path matrix shape {matrix.shape} != ({num_labels},{num_labels})")
    return torch.from_numpy(matrix)


def load_recall_reweight_vector(config, num_labels):
    path = config.get("recall_reweight_vector_path")
    if not path:
        return None
    payload = json.loads(Path(path).read_text())
    vector = np.asarray(payload["miss_rate"], dtype=np.float32)
    if vector.shape != (num_labels,):
        raise ValueError(f"recall_reweight_vector_path vector shape {vector.shape} != ({num_labels},)")
    return torch.from_numpy(vector)


def build_improved(config, labelmap, station_class_prior=None):
    id2label = {i: name for i, name in enumerate(labelmap.fine_names)}
    label2id = {name: i for i, name in enumerate(labelmap.fine_names)}
    if config.get("backbone") == "swin_small":
        from transformers import Mask2FormerForUniversalSegmentation
        base = Mask2FormerForUniversalSegmentation.from_pretrained(
            config["pretrained_checkpoint"], num_labels=31, id2label=id2label,
            label2id=label2id, ignore_mismatched_sizes=True)
    else:
        base = build_model(config)
    base.config.id2label = id2label; base.config.label2id = label2id
    matching_method = config.get("matching_method", "clinical_hungarian")
    if matching_method == "asymmetric_partial_sinkhorn":
        ot_options = {
            "epsilon": float(config.get("ot_epsilon", 0.5)),
            "iterations": int(config.get("ot_iterations", 120)),
            "target_mass_alpha": float(config.get("ot_target_mass_alpha", 0.5)),
            "target_mass_max": float(config.get("ot_target_mass_max", 2.0)),
            "dustbin_fp_weight": float(config.get("ot_dustbin_fp_weight", 0.1)),
            "pair_cost_risk_exponent": float(config.get("ot_pair_cost_risk_exponent", 0.0)),
            "dustbin_cost": float(config.get("ot_dustbin_cost", 0.0)),
            "mass_mode": str(config.get("ot_mass_mode", "fixed")),
            "convergence_tolerance": float(config.get("ot_convergence_tolerance", 0.0)),
            "check_interval": int(config.get("ot_sinkhorn_check_interval", 10)),
            "balance_iterations": int(config.get("ot_balance_iterations", 500)),
            "context_fp_strength": float(config.get("ot_context_fp_strength", 1.0)),
            "context_bound_mode": str(config.get("ot_context_bound_mode", "point")),
            "sink_margin_matrix": load_sink_margin_matrix(config, 31),
            "sink_margin_weight": float(config.get("sink_margin_weight", 0.0)),
            "sink_margin_value": float(config.get("sink_margin_value", 0.05)),
            "recall_reweight_vector": load_recall_reweight_vector(config, 31),
            "recall_reweight_weight": float(config.get("recall_reweight_weight", 0.0)),
        }
    else:
        ot_options = {"epsilon":float(config.get("ot_epsilon",0.08)),
                      "iterations":int(config.get("ot_iterations",40)),
                      "risk_exponent":float(config.get("ot_risk_exponent",1.0)),
                      "dustbin_cost":float(config.get("ot_dustbin_cost",0.0))}
    if bool(config.get("station_context", False)) and station_class_prior is None:
        station_class_prior = torch.ones(len(STATIONS), 31)
    if not bool(config.get("station_context", False)):
        station_class_prior = None
    model = ImprovedMask2Former(base, torch.from_numpy(labelmap.fine_to_coarse.astype(np.int64)),
                                torch.from_numpy(labelmap.fine_weights),
                                matching_method=matching_method,
                                ot_options=ot_options,
                                station_class_prior=station_class_prior,
                                context_samples=int(config.get("context_samples", 1)),
                                context_dropout=float(config.get("context_dropout", 0.0)))
    # from_pretrained may carry stale COCO metadata; persist TIGER's exact 31-class mapping.
    model.segmenter.config.id2label = {i: name for i, name in enumerate(labelmap.fine_names)}
    model.segmenter.config.label2id = {name: i for i, name in enumerate(labelmap.fine_names)}
    return model


def objective(result, batch, config, pos_weight, station_pos_weight,
              fine_class_weights=None):
    fine_loss = result["output"].loss
    coarse_weight = float(config["coarse_loss_weight"])
    hierarchy_weight = float(config["hierarchy_loss_weight"])
    presence_weight = float(config["presence_loss_weight"])
    context_weight = float(config.get("context_loss_weight", 0.0))
    cetl_weight = float(config.get("cetl_aux_loss_weight", 0.0))
    coarse_loss = fine_loss.new_zeros(()); hierarchy = fine_loss.new_zeros(())
    presence = fine_loss.new_zeros(())
    context = fine_loss.new_zeros(())
    cetl = fine_loss.new_zeros(()); cetl_ce = fine_loss.new_zeros(())
    cetl_tversky = fine_loss.new_zeros(())
    if coarse_weight > 0 or hierarchy_weight > 0:
        target_size = batch["fine"].shape[-2:]
        coarse_logits = F.interpolate(result["coarse_logits"], target_size, mode="bilinear",
                                      align_corners=False)
        if coarse_weight > 0:
            coarse_loss = F.cross_entropy(coarse_logits, batch["coarse"].to(coarse_logits.device))
        if hierarchy_weight > 0:
            direct = coarse_logits.softmax(1)
            fine = F.interpolate(result["fine_probability"], target_size, mode="bilinear",
                                 align_corners=False)
            derived = aggregate_fine_to_coarse(
                fine, result["output"].class_queries_logits.new_tensor(
                    config["fine_to_coarse"], dtype=torch.long))
            hierarchy = symmetric_kl(direct, derived)
    if presence_weight > 0:
        target_presence = presence_targets(batch["fine"].to(result["presence_logits"].device))
        presence = F.binary_cross_entropy_with_logits(
            result["presence_logits"], target_presence, pos_weight=pos_weight)
    if context_weight > 0:
        if result["station_logits"] is None:
            raise RuntimeError("context_loss_weight requires station_context")
        context = F.binary_cross_entropy_with_logits(
            result["station_logits"],
            batch["visibility"].to(result["station_logits"].device),
            pos_weight=station_pos_weight,
        )
    if cetl_weight > 0:
        weights = fine_class_weights if bool(
            config.get("cetl_use_clinical_class_weights", True)
        ) else None
        cetl, cetl_terms = cross_entropy_tversky_loss(
            result["fine_probability"],
            batch["fine"].to(result["fine_probability"].device),
            class_weights=weights,
            ce_fraction=float(config.get("cetl_ce_fraction", 0.5)),
            alpha=float(config.get("cetl_tversky_alpha", 0.3)),
            beta=float(config.get("cetl_tversky_beta", 0.7)),
            include_background=bool(config.get("cetl_include_background", True)),
            present_classes_only=bool(config.get("cetl_present_classes_only", True)),
            smooth=float(config.get("cetl_smooth", 1.0)),
        )
        cetl_ce = cetl_terms["cross_entropy"]
        cetl_tversky = cetl_terms["tversky"]
    total = (fine_loss + coarse_weight * coarse_loss + hierarchy_weight * hierarchy
             + presence_weight * presence + context_weight * context
             + cetl_weight * cetl)
    return total, {"fine": fine_loss, "coarse": coarse_loss, "hierarchy": hierarchy,
                   "presence": presence, "context": context, "cetl": cetl,
                   "cetl_ce": cetl_ce, "cetl_tversky": cetl_tversky}


@torch.no_grad()
def evaluate(model, loader, device, labelmap, config):
    model.eval(); predictions=[]; coarse_predictions=[]; targets=[]; cases=[]
    station_targets=[]; station_probabilities=[]
    task1_only = bool(config.get("task1_only", False))
    for batch in loader:
        result = model(batch["image"].to(device))
        fine = F.interpolate(result["fine_probability"], batch["fine"].shape[-2:], mode="bilinear", align_corners=False)
        if config.get("presence_head", True):
            fine = soft_presence_modulation(fine, result["presence_logits"], float(config["presence_epsilon"]),
                                            float(config["presence_gamma"]))
        predictions.extend(fine.argmax(1).cpu().numpy())
        if not task1_only:
            derived = aggregate_fine_to_coarse(fine, model.fine_to_coarse)
            if config.get("direct_coarse_head", True):
                direct = F.interpolate(result["coarse_logits"], batch["fine"].shape[-2:], mode="bilinear",
                                       align_corners=False).softmax(1)
                beta = float(config["coarse_fusion_beta"])
                fused = beta * direct + (1-beta) * derived
            else:
                fused = derived
            coarse_predictions.extend(fused.argmax(1).cpu().numpy())
        targets.extend(batch["fine"].numpy()); cases.extend(batch["case_id"])
        if result["station_logits"] is not None:
            station_targets.extend(batch["visibility"].numpy())
            station_probabilities.extend(result["station_logits"].sigmoid().cpu().numpy())
    fine_metric = segmentation_metrics(predictions, targets, cases, labelmap.fine_weights, labelmap.fine_names)
    failures = segmentation_failure_metrics(predictions, targets, labelmap.fine_names)
    task1_score = float(np.mean([fine_metric["dice"], 1-fine_metric["nhd"]]))
    output = {"task1_score": task1_score, "fine_dice": fine_metric["dice"],
              "fine_nhd": fine_metric["nhd"], "fine": fine_metric,
              "failure_metrics": failures}
    center_ids = sorted({case_id.split("_case_", 1)[0] for case_id in cases})
    fine_center_scores = []
    for center_id in center_ids:
        selected = {case_id for case_id in set(cases) if case_id.startswith(center_id + "_case_")}
        dice = float(np.mean([fine_metric["per_case_dice"][case_id] for case_id in selected]))
        nhd = float(np.mean([fine_metric["per_case_nhd"][case_id] for case_id in selected]))
        fine_center_scores.append((dice + 1.0 - nhd) / 2.0)
    output["center_macro_task1_score"] = float(np.mean(fine_center_scores))
    if station_targets:
        context_metric = visibility_metrics(
            np.asarray(station_targets), np.asarray(station_probabilities), STATIONS)
        output.update({"context_macro_f1": context_metric["macro_f1"],
                       "context_macro_auroc": context_metric["macro_auroc"],
                       "context": context_metric})
    if not task1_only:
        coarse_targets = [labelmap.fine_to_coarse[x] for x in targets]
        coarse_metric = segmentation_metrics(coarse_predictions, coarse_targets, cases,
                                             labelmap.coarse_weights, labelmap.coarse_names)
        surrogate = np.mean([fine_metric["dice"], 1-fine_metric["nhd"],
                             coarse_metric["dice"], 1-coarse_metric["nhd"]])
        output.update({"task12_surrogate": float(surrogate),
                       "coarse_dice": coarse_metric["dice"],
                       "coarse_nhd": coarse_metric["nhd"], "coarse": coarse_metric})
        center_surrogates = []
        for center_id in center_ids:
            selected = {case_id for case_id in set(cases) if case_id.startswith(center_id + "_case_")}
            values = (
                np.mean([fine_metric["per_case_dice"][case_id] for case_id in selected]),
                1.0 - np.mean([fine_metric["per_case_nhd"][case_id] for case_id in selected]),
                np.mean([coarse_metric["per_case_dice"][case_id] for case_id in selected]),
                1.0 - np.mean([coarse_metric["per_case_nhd"][case_id] for case_id in selected]),
            )
            center_surrogates.append(float(np.mean(values)))
        output["center_macro_task12_surrogate"] = float(np.mean(center_surrogates))
    return output


def run_fold(config, fold, args, labelmap, device, records_provider=records_for_fold):
    seed = int(config["seed"]) + fold; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    train_records = records_provider(args.data_root, fold, "train")
    train_ds = TigerMultitaskDataset(train_records, labelmap, config["height"], config["width"], True)
    val_ds = TigerMultitaskDataset(records_provider(args.data_root, fold, "validation"), labelmap,
                                   config["height"], config["width"], False)
    kwargs = {"num_workers": int(config["num_workers"]), "collate_fn": collate, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_ds, batch_size=int(config["batch_size"]), shuffle=True, **kwargs)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, **kwargs)
    context_prior = None
    if bool(config.get("station_context", False)):
        context_prior = station_anatomy_prior(
            train_records, labelmap, float(config.get("context_prior_smoothing", 1.0)))
    model = build_improved(config, labelmap, context_prior).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]),
                                  weight_decay=float(config["weight_decay"]))
    if float(config["presence_loss_weight"]) > 0:
        # Fold-specific imbalance must be measured on original masks, never on random augmentation.
        presence_ds = TigerMultitaskDataset(
            train_records, labelmap, config["height"], config["width"], False)
        all_presence = torch.stack(
            [presence_targets(item["fine"][None])[0] for item in presence_ds]).to(device)
        pos_weight = presence_pos_weight(
            all_presence, float(config.get("presence_pos_weight_max", 8)))
    else:
        pos_weight = torch.ones(31, device=device)
    if float(config.get("context_loss_weight", 0.0)) > 0:
        station_pos_weight = torch.from_numpy(visibility_pos_weight(train_records)).to(device)
    else:
        station_pos_weight = torch.ones(len(STATIONS), device=device)
    fine_class_weights = torch.from_numpy(labelmap.fine_weights).to(device)
    out = args.output_root / config["experiment_id"] / f"fold_{fold}"; out.mkdir(parents=True, exist_ok=True)
    epochs = int(config.get("smoke_epochs", 1)) if args.smoke else int(config["epochs"]); history=[]; best=-float("inf"); bad_epochs=0
    patience=int(config["early_stopping_patience"]); delta=float(config["early_stopping_min_delta"])
    for epoch in range(epochs):
        model.train(); optimizer.zero_grad(set_to_none=True); running=[]; ot_diagnostics=[]
        running_terms = {}
        for step, batch in enumerate(train_loader):
            result = model(batch["image"].to(device), [x.to(device) for x in batch["mask_labels"]],
                           [x.to(device) for x in batch["class_labels"]])
            diagnostics = getattr(model.segmenter.criterion, "last_diagnostics", None)
            if diagnostics:
                ot_diagnostics.append(dict(diagnostics))
            if (float(config["coarse_loss_weight"]) > 0
                    or float(config["hierarchy_loss_weight"]) > 0):
                batch["coarse"] = torch.from_numpy(
                    labelmap.fine_to_coarse[batch["fine"].numpy()]).long()
            total, terms = objective(
                result, batch, config, pos_weight, station_pos_weight,
                fine_class_weights=fine_class_weights,
            )
            for name, value in terms.items():
                running_terms.setdefault(name, []).append(float(value.detach()))
            (total / int(config["gradient_accumulation"])).backward(); running.append(float(total.detach()))
            if (step+1) % int(config["gradient_accumulation"]) == 0 or step+1 == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step(); optimizer.zero_grad(set_to_none=True)
        metric = evaluate(model, val_loader, device, labelmap, config)
        compact = {key:value for key,value in metric.items() if isinstance(value, float)}
        row={"epoch":epoch,"train_loss":float(np.mean(running)),**{f"val_{k}":v for k,v in compact.items()}}
        row.update({f"train_loss_{name}":float(np.mean(values))
                    for name, values in running_terms.items()})
        if ot_diagnostics:
            scalar_keys = [key for key, value in ot_diagnostics[0].items()
                           if isinstance(value, (int, float))]
            row.update({f"train_ot_{key}":float(np.mean([item[key] for item in ot_diagnostics]))
                        for key in scalar_keys})
        history.append(row); print(json.dumps(row), flush=True)
        selection_metric = str(config.get("selection_metric", "task12_surrogate"))
        if selection_metric not in metric:
            raise KeyError(f"Unknown selection_metric {selection_metric!r}; available={sorted(metric)}")
        score=metric[selection_metric]
        if score > best + delta:
            best=score; bad_epochs=0
            torch.save({"model":model.state_dict(),"config":config,"epoch":epoch,"metrics":row}, out/"best.pt")
            (out/"best_validation_metrics.json").write_text(json.dumps(metric,indent=2)+"\n")
        else: bad_epochs += 1
        if bad_epochs >= patience: break
    (out/"epoch_metrics.json").write_text(json.dumps(history,indent=2)+"\n")
    selection_metric = str(config.get("selection_metric", "task12_surrogate"))
    (out/"early_stopping.json").write_text(json.dumps({"monitor":f"val_{selection_metric}","mode":"max",
        "patience":patience,"min_delta":delta,"best":best,"stopped_epoch":history[-1]["epoch"],
        "bad_epochs":bad_epochs},indent=2)+"\n")
    (out/"completion.json").write_text(json.dumps({"status":"smoke" if args.smoke else "completed",
        "fold":fold,"effective_seed":seed,"epochs_run":len(history)},indent=2)+"\n")


def main():
    args=parse_args(); config=yaml.safe_load(args.config.read_text()); labelmap=LabelMap.load(args.data_root/"labelmap.csv")
    config["fine_to_coarse"] = labelmap.fine_to_coarse.astype(int).tolist()
    if not torch.cuda.is_available() and not args.allow_cpu: raise RuntimeError("CUDA unavailable; use --allow-cpu for smoke")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for fold in (range(5) if args.run_all_folds else [args.fold]): run_fold(config,fold,args,labelmap,device)

if __name__ == "__main__": main()
