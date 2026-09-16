#!/usr/bin/env python3
"""Five-fold center-macro Task-1 comparison: baseline coarse-only, aug, geo."""
import argparse
import json
from pathlib import Path

import numpy as np


def summary(root, arm):
    rows = []
    for fold in range(5):
        m = json.loads((root / arm / f"fold_{fold}/validation.json").read_text())["metrics"]
        for case, d in m["per_case_dice"].items():
            n = m["per_case_nhd"][case]
            rows.append({"case": case, "fold": fold, "dice": d, "nhd": n, "score": (d + 1 - n) / 2})
    assert len({r["case"] for r in rows}) == len(rows) == 40
    centers = {}
    for c in sorted({r["case"].split("_case_")[0] for r in rows}):
        s = [r for r in rows if r["case"].startswith(c + "_case_")]
        d, n = np.mean([r["dice"] for r in s]), np.mean([r["nhd"] for r in s])
        centers[c] = {"dice": float(d), "nhd": float(n), "score": float((d + 1 - n) / 2)}
    cd = float(np.mean([v["dice"] for v in centers.values()]))
    cn = float(np.mean([v["nhd"] for v in centers.values()]))
    return {"center_macro": {"dice": cd, "nhd": cn, "score": (cd + 1 - cn) / 2},
            "case_macro_score": float(np.mean([r["score"] for r in rows])),
            "worst_center_score": min(v["score"] for v in centers.values()),
            "per_center": centers, "per_case": {r["case"]: r for r in rows}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    root = p.parse_args().root
    out = {arm: summary(root, arm) for arm in ("baseline", "aug", "geo")}
    base = out["baseline"]["per_case"]
    for arm in ("aug", "geo"):
        diff = [out[arm]["per_case"][c]["score"] - base[c]["score"] for c in base]
        out[arm]["vs_baseline"] = {"center_macro_score": out[arm]["center_macro"]["score"] - out["baseline"]["center_macro"]["score"],
                                   "cases_better": int(sum(x > 0 for x in diff)), "cases_worse": int(sum(x < 0 for x in diff))}
    g, a = out["geo"]["per_case"], out["aug"]["per_case"]
    out["geo"]["vs_aug"] = {"center_macro_score": out["geo"]["center_macro"]["score"] - out["aug"]["center_macro"]["score"],
                            "cases_better": int(sum(g[c]["score"] > a[c]["score"] for c in g)),
                            "cases_worse": int(sum(g[c]["score"] < a[c]["score"] for c in g))}
    (root / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
    for arm, s in out.items():
        print(arm, json.dumps({k: v for k, v in s.items() if k not in ("per_case", "per_center")}))


if __name__ == "__main__":
    main()
