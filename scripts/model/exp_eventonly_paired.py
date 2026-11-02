"""Like-for-like comparison: the production five-seed ensemble (58 inputs) against the event-only model (14 inputs) on the SAME test seconds.

The two saved test sets differ (the production model is scored on the 3,059 2026 test matches; the event-only metadata reports 3,601 matches
across both years), so their headline numbers are not comparable. This script scores the event-only booster on the 2026 test seconds that
the production export holds, and reports the paired difference (event-only minus production) per second, with a match-cluster 95% interval
(1,000 resamples), overall and by round group.

  ...python.exe -I -X utf8 scripts\\model\\exp_eventonly_paired.py
Writes data/combined/exp_eventonly_paired.json.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import polars as pl
import xgboost as xgb

import wp_common as W

sys.stdout.reconfigure(encoding="utf-8")
N_BOOT = 1000


def logloss(y, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def match_bootstrap(keys, d, n=N_BOOT, seed=0):
    _, inv = np.unique(np.asarray(keys), return_inverse=True)
    s = np.bincount(inv, weights=np.asarray(d, dtype=np.float64))
    c = np.bincount(inv).astype(np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(n, len(s)))
    means = s[idx].sum(axis=1) / c[idx].sum(axis=1)
    return float(s.sum() / c.sum()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    d = os.path.join(W.ROOT, "data", "combined")
    meta = json.load(open(os.path.join(W.ROOT, "models", "winprob_v7_basic_scorelead.meta.json"), encoding="utf-8"))
    feats = meta["features"]
    booster = xgb.Booster()
    booster.load_model(os.path.join(W.ROOT, "models", "winprob_v7_basic_scorelead.json"))
    best = int(meta["best_iteration"])

    ex = pl.read_parquet(os.path.join(d, "export_winprob_v8_ensemble_test_predictions.parquet"))
    print(f"production test seconds {ex.height:,} on {ex['match_key'].n_unique():,} matches", flush=True)
    base = [f for f in feats if not f.startswith("map_is_")]
    test_keys = ex["match_key"].unique().to_list()
    grid = (pl.scan_parquet(os.path.join(d, "grid.parquet")).select(["match_key", "round", "tick", "label_ct_win", "map_name"] + base)
            .filter(pl.col("match_key").is_in(test_keys)).collect())
    # the seven map indicators, built from the map name the same way the training rows were (out-of-scope maps are all zero)
    maps = [f[len("map_is_"):] for f in feats if f.startswith("map_is_")]
    grid = grid.with_columns([(pl.col("map_name") == f"de_{m}").cast(pl.Float32).alias(f"map_is_{m}") for m in maps])
    print("map_name in the grid, top values:", grid["map_name"].value_counts().sort("count", descending=True).head(9).to_dicts(), flush=True)
    j = ex.join(grid, on=["match_key", "round", "tick"], how="left")
    miss = j.filter(pl.col("label_ct_win").is_null()).height
    print(f"joined rows {j.height:,}; rows without a grid match {miss}", flush=True)
    if miss:
        raise SystemExit("join incomplete: stop rather than compare different seconds")
    X = j.select(feats).to_numpy().astype(np.float32)
    y = j["label_ct_win"].to_numpy().astype(np.float64)
    p_prod = j["p"].to_numpy().astype(np.float64)
    p_ev = booster.predict(xgb.DMatrix(X, feature_names=feats), iteration_range=(0, best + 1)).astype(np.float64)
    keys = j["match_key"].to_numpy()
    rnd = j["round"].to_numpy()

    d_loss = logloss(y, p_ev) - logloss(y, p_prod)
    mean, lo, hi = match_bootstrap(keys, d_loss)
    out = {"test_seconds": int(j.height), "test_matches": int(len(np.unique(keys))),
           "production_test_logloss": float(logloss(y, p_prod).mean()), "event_only_test_logloss": float(logloss(y, p_ev).mean()),
           "event_only_minus_production": mean, "ci95": [lo, hi], "n_boot": N_BOOT, "event_only_best_iteration": best, "groups": {}}
    groups = [("pistol (1, 13)", np.isin(rnd, [1, 13])), ("after pistol (2, 14)", np.isin(rnd, [2, 14])),
              ("other rounds", ~np.isin(rnd, [1, 2, 13, 14]))]
    for name, m in groups:
        gm, glo, ghi = match_bootstrap(keys[m], d_loss[m])
        out["groups"][name] = {"seconds": int(m.sum()), "event_only_minus_production": gm, "ci95": [glo, ghi]}
        print(f"{name}: {m.sum():,} seconds; event-only minus production {gm:+.6f} [{glo:+.6f}, {ghi:+.6f}]", flush=True)
    print(f"overall: production {out['production_test_logloss']:.6f}; event-only {out['event_only_test_logloss']:.6f}; "
          f"event-only minus production {mean:+.6f} [{lo:+.6f}, {hi:+.6f}]", flush=True)
    W.dump_json(out, os.path.join(d, "exp_eventonly_paired.json"))


if __name__ == "__main__":
    main()
