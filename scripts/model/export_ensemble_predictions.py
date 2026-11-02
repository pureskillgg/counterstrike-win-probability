"""Grid predictions of winprob_v8_ensemble (mean of the five seed models) on the frozen test split, for the serving parity test.

Features are the cache's 58 inputs (59 minus heavy, plus score lead), the same rows as the reported test score. Writes
data/combined/export_winprob_v8_ensemble_test_predictions.parquet with match_key, round, tick, p (CT win probability, 0 to 1).

  ...python.exe -I -X utf8 scripts\\model\\export_ensemble_predictions.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import polars as pl
import xgboost as xgb

import wp_common as W

sys.stdout.reconfigure(encoding="utf-8")
HEAVY = ["st_heavy_ct", "st_heavy_t"]


def main():
    t0 = time.time()
    d = os.path.join(W.ROOT, "data", "combined")
    meta = json.load(open(os.path.join(W.ROOT, "models", "winprob_v8_ensemble.meta.json"), encoding="utf-8"))
    feats = meta["features"]
    feats59 = [f for f in feats if f != "score_lead_diag"]
    need = list(dict.fromkeys(["match_key", "round", "tick", "second", "label_ct_win", "w_round"] + feats59))
    G = pl.scan_parquet(os.path.join(d, "cache_snapshot_2026.parquet")).select(need).with_columns(pl.col(feats59).cast(pl.Float32)).collect()
    lead = (pl.scan_parquet(os.path.join(d, "grid.parquet")).filter(pl.col("era") == "2026").select(["match_key", "round", "score_lead_diag"])
            .unique(subset=["match_key", "round"]).collect())
    G = G.join(lead, on=["match_key", "round"], how="left").with_columns(pl.col("score_lead_diag").fill_null(0).cast(pl.Float32))
    kdf = G.select("match_key").unique().with_columns(pl.lit(1.0).alias("w_round"), pl.lit(0).alias("label_ct_win"),
                                                      pl.lit("2026").alias("era"), pl.lit("").alias("match_date"))
    sp = W.make_split("random", kdf, seed=0)
    te = sorted(sp["test"])
    sub, X, y = W.arrays(G, te, feats, stride=1)
    dm = xgb.DMatrix(X)
    preds = []
    for m in meta["members"]:
        b = xgb.Booster()
        b.load_model(os.path.join(W.ROOT, "models", f"winprob_v8_seed{m['seed']}.json"))
        preds.append(b.predict(dm, iteration_range=(0, int(m["best_iteration"]) + 1)))
    p = np.mean(preds, axis=0)
    out = sub.select(["match_key", "round", "tick"]).with_columns(pl.Series("p", p))
    out.write_parquet(os.path.join(d, "export_winprob_v8_ensemble_test_predictions.parquet"))
    print(f"wrote {out.height:,} test seconds ({out['match_key'].n_unique():,} matches) ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
