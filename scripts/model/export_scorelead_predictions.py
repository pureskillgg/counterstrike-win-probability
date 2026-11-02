"""Grid predictions of winprob_v7_basic_scorelead on the frozen test split (every second of every test match), for the serving parity test.

Features are built from the grid exactly as in training (train_final.load_basic, roster-defect rounds dropped, plus score_lead_diag from the
grid). Writes data/combined/export_winprob_v7_basic_scorelead_test_predictions.parquet with match_key, round, tick, p (CT win probability, 0 to 1).

  ...python.exe -I -X utf8 scripts\\model\\export_scorelead_predictions.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import polars as pl
import xgboost as xgb

import train_final as T
import wp_common as W

sys.stdout.reconfigure(encoding="utf-8")


def main():
    t0 = time.time()
    d = os.path.join(W.ROOT, "data", "combined")
    meta = json.load(open(os.path.join(W.ROOT, "models", "winprob_v7_basic_scorelead.meta.json"), encoding="utf-8"))
    feats = meta["features"]
    G, feats_b = T.load_basic("combined")
    lead = (pl.scan_parquet(os.path.join(d, "grid.parquet")).select(["match_key", "round", "score_lead_diag"])
            .unique(subset=["match_key", "round"]).collect())
    G = G.join(lead, on=["match_key", "round"], how="left").with_columns(pl.col("score_lead_diag").fill_null(0).cast(pl.Float32))
    assert feats[:-1] == feats_b and feats[-1] == "score_lead_diag", "feature order differs from the basic model + score lead"
    sp = W.make_split("random", G, seed=0)
    te = sorted(sp["test"])
    sub, X, y = W.arrays(G, te, feats, stride=1)
    b = xgb.Booster()
    b.load_model(os.path.join(W.ROOT, "models", "winprob_v7_basic_scorelead.json"))
    p = b.predict(xgb.DMatrix(X), iteration_range=(0, int(meta["best_iteration"]) + 1))
    out = sub.select(["match_key", "round", "tick"]).with_columns(pl.Series("p", p))
    out.write_parquet(os.path.join(d, "export_winprob_v7_basic_scorelead_test_predictions.parquet"))
    print(f"wrote {out.height:,} test seconds ({out['match_key'].n_unique():,} matches) ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
