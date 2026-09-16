#!/usr/bin/env python3
"""Build a leakage-safe class confusion graph from fold validation artifacts."""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tiger_models.data import LabelMap


def build_graph(fold_metrics: list[dict], labelmap: LabelMap, top_k: int) -> dict:
    matrices = [np.asarray(item["failure_metrics"]["confusion_pixels"], dtype=np.int64)
                for item in fold_metrics]
    if any(matrix.shape != (31, 31) for matrix in matrices):
        raise ValueError("Every confusion matrix must be 31x31")
    confusion = np.stack(matrices).sum(0)
    support = confusion.sum(1)
    directed_rate = confusion / np.maximum(support[:, None], 1)
    np.fill_diagonal(directed_rate, 0.)
    # Symmetric neighbor strength retains the larger directional failure rate;
    # direction-specific rates/counts remain attached to every edge.
    symmetric = np.maximum(directed_rate, directed_rate.T)
    candidates=[]
    for first in range(1,31):
        for second in range(first+1,31):
            candidates.append({
                "class_i":first,"name_i":labelmap.fine_names[first],
                "class_j":second,"name_j":labelmap.fine_names[second],
                "edge_weight":float(symmetric[first,second]),
                "i_to_j_rate":float(directed_rate[first,second]),
                "j_to_i_rate":float(directed_rate[second,first]),
                "i_to_j_pixels":int(confusion[first,second]),
                "j_to_i_pixels":int(confusion[second,first]),
                "clinical_risk":float(max(labelmap.fine_weights[first],labelmap.fine_weights[second])),
            })
    candidates.sort(key=lambda item:(item["edge_weight"],
                                      item["i_to_j_pixels"]+item["j_to_i_pixels"]),reverse=True)
    edges=[item for item in candidates if item["edge_weight"]>0][:top_k]
    return {"nodes":[{"class_id":i,"class_name":name,"clinical_weight":float(labelmap.fine_weights[i]),
                       "pixel_support":int(support[i])} for i,name in enumerate(labelmap.fine_names)],
            "edges":edges,"aggregate_confusion_pixels":confusion.tolist()}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model-dir",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--data-root",type=Path,default=ROOT/"data")
    parser.add_argument("--top-k",type=int,default=20)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(f"Refusing to overwrite {args.output}")
    paths=[args.model_dir/f"fold_{fold}"/"best_validation_metrics.json" for fold in range(5)]
    missing=[str(path) for path in paths if not path.is_file()]
    if missing: raise FileNotFoundError(f"Missing validation artifacts: {missing}")
    metrics=[json.loads(path.read_text()) for path in paths]
    labelmap=LabelMap.load(args.data_root/"labelmap.csv")
    graph=build_graph(metrics,labelmap,args.top_k)
    payload={"split":"out_of_fold_validation_only","test_data_used":False,
             "source_model_dir":str(args.model_dir),"source_files":[str(path) for path in paths],
             "normalization":"directed error pixels / ground-truth source-class pixels",
             "symmetric_edge_weight":"max(i_to_j_rate, j_to_i_rate)",**graph}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps({"output":str(args.output),"edges":payload["edges"]},indent=2))

if __name__=="__main__": main()
