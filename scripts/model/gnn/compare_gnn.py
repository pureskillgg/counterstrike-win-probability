"""Paired comparison of the graph / set models with the tabular references on the same test rows.   (CPU)

  ...python.exe -I -X utf8 scripts\\model\\gnn\\compare_gnn.py [--refs xgb59,xgb83] [--models name1,name2,...]

Reads data/gnn/pred_<name>.npy (+ _rows.npy) of every model, checks that all use the same test rows, and scores each against each reference with the match-cluster paired
bootstrap of wp_common.evaluate (the same code as every other comparison in the project). Writes data/combined/exp_gnn.json.
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import numpy as np
import polars as pl

import wp_common as W

sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"G:\pureskill\csds-cs2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", default="xgb59,xgb83")
    ap.add_argument("--models", default="")
    a = ap.parse_args()
    g = os.path.join(ROOT, "data", "gnn")
    idx = pl.read_parquet(os.path.join(g, "index.parquet"))
    names = [n for n in a.models.split(",") if n] or sorted({os.path.basename(f)[5:-4] for f in glob.glob(os.path.join(g, "pred_*.npy")) if not f.endswith("_rows.npy")})
    refs = [r for r in a.refs.split(",") if r]
    rows0 = None
    P = {}
    for n in names:
        rows = np.load(os.path.join(g, f"pred_{n}_rows.npy"))
        if rows0 is None:
            rows0 = rows
        assert np.array_equal(rows, rows0), f"{n} was scored on different rows"
        P[n] = np.load(os.path.join(g, f"pred_{n}.npy")).astype(np.float64)
    sub = idx[rows0].select("match_key", "round", "tick", "w_round")
    y = idx["label_ct_win"].to_numpy()[rows0].astype(np.int64)
    keep = ("logloss", "logloss_ci95", "auc", "ece10", "ece10_ci95")
    out = {"test_rows": int(len(rows0)), "test_matches": int(sub["match_key"].n_unique()), "models": {}}
    for n in names:
        r = W.evaluate(sub, y, P[n])
        row = {k: r[k] for k in keep}
        for ref in refs:
            if ref in P and ref != n:
                rr = W.evaluate(sub, y, P[n], ref=P[ref])
                row[f"delta_vs_{ref}"], row[f"delta_vs_{ref}_ci95"] = rr["delta_vs_ref"], rr["delta_vs_ref_ci95"]
        res = os.path.join(g, f"result_{n}.json")
        if os.path.exists(res):
            j = json.load(open(res, encoding="utf-8"))
            row["parameters"] = j.get("parameters")
            row["args"] = j.get("args")
            row["train_seconds"] = j.get("train_seconds")
        out["models"][n] = row
        print(f"{n}: log loss {r['logloss']:.4f} {[round(x, 4) for x in r['logloss_ci95']]} AUC {r['auc']:.4f} ECE {r['ece10']:.4f}" +
              "".join(f"  vs {ref} {row[f'delta_vs_{ref}']:+.4f} {[round(x, 4) for x in row[f'delta_vs_{ref}_ci95']]}" for ref in refs if f"delta_vs_{ref}" in row), flush=True)
    W.dump_json(out, os.path.join(ROOT, "data", "combined", "exp_gnn.json"))


if __name__ == "__main__":
    main()
