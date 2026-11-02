"""Paired comparison: all-maps GRU with straight-line distance features against the same model without them, seed by seed.

Pre-registered (written before the run; also in train_temporal_allmaps.py): the distance model beats the baseline if the mean over seeds
0, 1 and 2 of the paired log-loss difference (distance minus baseline) is <= -0.0005 and every seed's 95% match-cluster CI upper bound
is < 0. Paired on the de_mirage test seconds (spatial_feature_eval.paired, 1,000 resamples, seed 0).

Run in the cs2gpu environment (polars, numpy):
  C:/Users/13144/.pixi/workspace-envs/cs2gpu-15282324412032665829/envs/default/python.exe scripts/model/allmaps_dist_compare.py
Writes data/temporal/paired_allmaps_dist_vs_baseline.json.
"""
import argparse
import json
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
from spatial_feature_eval import paired  # noqa: E402

ROOT = "G:/pureskill/csds-cs2"
T = os.path.join(ROOT, "data", "temporal")
OUT = os.path.join(T, "paired_allmaps_dist_vs_baseline.json")
KEY = ["match_key", "round", "second"]
SEEDS = [0, 1, 2]


def preds(name):
    return pl.read_parquet(os.path.join(T, name)).sort(KEY)


def run_name(kind, s):
    """Prediction file stem for a model kind ('baseline', 'dist', 'walk') and seed."""
    suffix = "" if s == 0 else f"_seed{s}"
    return f"allmaps_gru_loadout{'' if kind == 'baseline' else '_' + kind}{suffix}_preds.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="dist", choices=["dist", "walk"])
    ap.add_argument("--reference", default="baseline", choices=["baseline", "dist"])
    a = ap.parse_args()
    per_seed = {}
    for s in SEEDS:
        base = preds(run_name(a.reference, s))
        dist = preds(run_name(a.variant, s))
        if base.height != dist.height or not (base.select(KEY).equals(dist.select(KEY))):
            raise SystemExit(f"seed {s}: test rows differ between the two runs")
        y = base["label"].to_numpy().astype(np.int64)
        match = base["match_key"].to_numpy()
        per_seed[s] = paired(match, y, dist["p"].to_numpy(), base["p"].to_numpy())
        print(f"seed {s}: {a.variant} minus {a.reference} {per_seed[s]['diff']:+.6f}, CI {per_seed[s]['ci95']}", flush=True)
    d = [per_seed[s]["diff"] for s in SEEDS]
    ci = [per_seed[s]["ci95"] for s in SEEDS]
    mean = float(np.mean(d))
    res = {"variant": a.variant, "reference": a.reference, "rows": int(base.height), "per_seed": {str(s): per_seed[s] for s in SEEDS},
           "mean_over_seeds": mean, "passes": bool(mean <= -0.0005 and all(c[1] < 0 for c in ci)),
           "rule": "mean <= -0.0005 and every seed's CI upper bound < 0"}
    os.makedirs(T, exist_ok=True)
    out = os.path.join(T, f"paired_allmaps_{a.variant}_vs_{a.reference}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({"variant": a.variant, "reference": a.reference, "mean_over_seeds": mean, "passes": res["passes"]}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
