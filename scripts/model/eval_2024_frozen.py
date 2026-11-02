"""Out-of-time check: the frozen 57-input XGBoost (trained on 2026) scored on the 2024 February rounds that survive the data rules.

Rows: data/combined/cache_snapshot_2024.parquet (make_snapshot_cache.py --era 2024). That builder already leaves out roster-defect rounds
and the rounds whose status table disagrees with the event counts (features_status.inconsistent_rounds). This script also applies the
rule agreed for 2024: a round with a death whose victim side is null in deaths.parquet is excluded (those deaths cannot be counted toward
a side, so the alive counts are wrong for that round). It reports how many of the excluded rounds the builder had already dropped.

Metrics: log loss, accuracy and AUC on the kept rounds, overall and per map; log-loss intervals by match-cluster bootstrap (1,000 resamples,
seed 0). Nothing is tuned: the model is the saved frozen booster and its saved feature list. The 2024 numbers are descriptive (no
paired reference), and the reference numbers from 2026 are quoted only for scale.

Writes data/diagnostics/eval_2024_frozen.json.
Run in the cs2gpu environment (polars, xgboost):
  C:/Users/13144/.pixi/workspace-envs/cs2gpu-15282324412032665829/envs/default/python.exe scripts/model/eval_2024_frozen.py
"""
import glob
import json
import os
import sys

import numpy as np
import polars as pl
import xgboost as xgb

sys.path.insert(0, os.path.dirname(__file__))
import wp_common as W  # noqa: E402

ROOT = W.ROOT
OUT = os.path.join(ROOT, "data", "diagnostics", "eval_2024_frozen.json")
KEY = ["match_key", "round"]


def bootstrap_ci(match, loss, n=1000, seed=0):
    uniq, inv = np.unique(match, return_inverse=True)
    sums = np.bincount(inv, weights=loss, minlength=uniq.size)
    cnts = np.bincount(inv, minlength=uniq.size).astype(np.float64)
    rng = np.random.default_rng(seed)
    boot = np.empty(n)
    for i in range(n):
        pick = rng.integers(0, uniq.size, uniq.size)
        boot[i] = sums[pick].sum() / cnts[pick].sum()
    return [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


def metrics(df, feats, booster):
    X = xgb.DMatrix(df.select(feats).to_numpy().astype(np.float32), feature_names=feats)
    p = booster.predict(X)
    y = df["label_ct_win"].to_numpy().astype(np.int64)
    ll = W.ll_vec(p, y)
    return {"rounds": int(df.select(KEY).unique().height), "matches": int(df["match_key"].n_unique()), "rows": int(df.height),
            "log_loss": float(ll.mean()), "log_loss_ci95_match_bootstrap": bootstrap_ci(df["match_key"].to_numpy(), ll),
            "accuracy": float(((p >= 0.5).astype(np.int64) == y).mean()), "auc": float(W.auc(y, p)),
            "ct_win_rate": float(y.mean())}, p


def main():
    feats = json.load(open(os.path.join(ROOT, "models", "winprob_v6_drop_heavy.meta.json"), encoding="utf-8"))["features"]
    assert len(feats) == 57, len(feats)
    booster = xgb.Booster()
    booster.load_model(os.path.join(ROOT, "models", "winprob_v6_drop_heavy.json"))
    C = pl.read_parquet(os.path.join(ROOT, "data", "combined", "cache_snapshot_2024.parquet"))
    C = C.filter(pl.col("label_ct_win").is_not_null())
    print(f"2024 cache rows {C.height:,}, rounds {C.select(KEY).unique().height:,}", flush=True)

    deaths = pl.read_parquet(os.path.join(ROOT, "data", "combined", "deaths.parquet"), columns=["match_key", "round", "side"])
    null_rounds = deaths.filter(pl.col("side").is_null()).select(["match_key", pl.col("round").cast(pl.Int64)]).unique()
    C = C.with_columns(pl.col("round").cast(pl.Int64))
    # Matches with no status file (extraction failed on the corrupt raw file) have no loadout features: excluded by rule.
    status_matches = set()
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "status_1hz_2024", "*.parquet"))):
        status_matches.update(pl.read_parquet(f, columns=["match_key"])["match_key"].unique().to_list())
    all_matches = C["match_key"].unique().to_list()
    no_status = [m for m in all_matches if m not in status_matches]
    C = C.filter(pl.col("match_key").is_in(list(status_matches)))
    in_cache_null = C.select(KEY).unique().join(null_rounds, on=KEY, how="inner").height
    kept = C.join(null_rounds.with_columns(pl.lit(True).alias("_null")), on=KEY, how="left").filter(pl.col("_null").is_null())
    summary = {"matches_in_cache": len(all_matches), "matches_without_status_file_excluded": len(no_status),
               "rounds_before_null_rule": C.select(KEY).unique().height,
               "rounds_with_null_side_death": int(in_cache_null),
               "rounds_kept": kept.select(KEY).unique().height}

    overall, _ = metrics(kept, feats, booster)
    per_map = {}
    for mp in sorted(kept["map_name"].unique().to_list()):
        per_map[mp], _ = metrics(kept.filter(pl.col("map_name") == mp), feats, booster)
    res = {"model": "winprob_v6_drop_heavy (57 inputs, trained on 2026)", "rule": "roster-defect and inconsistent rounds (builder) "
           "plus rounds with a null-side death (this script)", "data_rules": summary, "overall": overall, "per_map": per_map}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({"data_rules": summary, "overall": overall}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
