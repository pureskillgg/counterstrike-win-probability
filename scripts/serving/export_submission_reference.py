"""Reference predictions for the 57-input submission (winprob_v6_drop_heavy) on the 2026 test rows, for the parity test.

  ...python.exe -I -X utf8 scripts\\serving\\export_submission_reference.py

Scores the saved booster on the cache rows of the frozen test split (the same rows as scripts/model/exp_submission_57.py) and writes
data/combined/export_winprob_v6_drop_heavy_test_predictions.parquet with match_key, round, tick and p (a fraction, 0 to 1).
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))
import polars as pl
import xgboost as xgb

import wp_common as W

sys.stdout.reconfigure(encoding="utf-8")


def main():
    d = os.path.join(W.ROOT, "data", "combined")
    meta = json.load(open(os.path.join(W.ROOT, "models", "winprob_v6_drop_heavy.meta.json"), encoding="utf-8"))
    feats = meta["features"]
    bst = xgb.Booster()
    bst.load_model(os.path.join(W.ROOT, "models", "winprob_v6_drop_heavy.json"))
    best = int(meta["best_iteration"])

    man = json.load(open(os.path.join(d, "split_manifest.json"), encoding="utf-8"))["assign"]
    cols = ["match_key", "round", "tick"] + feats
    cache = pl.read_parquet(os.path.join(d, "cache_snapshot_2026.parquet"), columns=cols)
    cache = cache.filter(pl.col("match_key").str.starts_with("csds/2026/"))
    cache = cache.with_columns(pl.col("match_key").map_elements(lambda k: man.get(k, ""), return_dtype=pl.Utf8).alias("split"))
    test = cache.filter(pl.col("split") == "test")
    X = test.select(feats).to_numpy().astype("float32")
    p = bst.predict(xgb.DMatrix(X, feature_names=feats), iteration_range=(0, best + 1))
    out = test.select(["match_key", "round", "tick"]).with_columns(pl.Series("p", p.astype("float64")))
    path = os.path.join(d, "export_winprob_v6_drop_heavy_test_predictions.parquet")
    out.write_parquet(path)
    print(f"wrote {out.height:,} test rows ({out['match_key'].n_unique():,} matches) to {path}")


if __name__ == "__main__":
    main()
