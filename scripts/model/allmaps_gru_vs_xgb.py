"""Matched-scope comparison: the all-maps GRU against the all-maps XGBoost reference, on the same frozen de_mirage test seconds.

Both models are trained on every map's train split (extended data) and early-stopped on every map's validation split, and both
are scored on the 687 frozen de_mirage test matches. This is the comparison that the earlier single-map runs could not make.

Pairs (diff = a - b; negative means a is better), match-cluster bootstrap, 1000 resamples, seed 0:
  all-maps GRU vs all-maps XGBoost (extended data)      <- the question
  all-maps GRU vs all-maps XGBoost (original data)      <- the GRU is only on extended data, so this is context, not a pair
  all-maps XGBoost, extended vs original                <- what the extra matches did to the reference

  ...torch env python.exe -I -X utf8 -u scripts\\model\\allmaps_gru_vs_xgb.py
Writes data/diagnostics/allmaps_gru_vs_xgb.json.
"""
import json
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
import wp_common as W  # noqa: E402
import ext_temporal_paired as etp  # noqa: E402

ROOT = W.ROOT
DIAG = os.path.join(ROOT, "data", "diagnostics")
TEMPORAL = os.path.join(ROOT, "data", "temporal")
OUT = os.path.join(DIAG, "allmaps_gru_vs_xgb.json")
KEY = ["match_key", "round", "second"]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="allmaps_gru_loadout", help="prediction tag in data/temporal (e.g. allmaps_gru_loadout_seed1)")
    a = ap.parse_args()
    out = os.path.join(DIAG, f"allmaps_gru_vs_xgb_{a.tag}.json")
    gru = pl.read_parquet(os.path.join(TEMPORAL, f"{a.tag}_preds.parquet")).select(KEY + ["label", pl.col("p").alias("gru")])
    xgb = pl.read_parquet(os.path.join(DIAG, "ext_baseline_all_preds.parquet"))
    xgb = xgb.select(KEY + ["label", pl.col("p_E").alias("xgb_E"), pl.col("p_O").alias("xgb_O")])
    J = gru.join(xgb, on=KEY, how="inner", suffix="_x")
    if J.height != gru.height or J.height != xgb.height:
        raise SystemExit(f"test rows do not pair: GRU {gru.height}, XGB {xgb.height}, joined {J.height}")
    if not (J["label"] == J["label_x"]).all():
        raise SystemExit("labels disagree on paired rows")
    y = J["label"].to_numpy().astype(np.int64)
    match = J["match_key"].to_numpy()

    ll = {c: float(W.ll_vec(J[c].to_numpy(), y).mean()) for c in ["gru", "xgb_E", "xgb_O"]}
    pairs = {
        "gru_vs_xgb_all_extended": etp.paired(match, y, J["gru"].to_numpy(), J["xgb_E"].to_numpy()),
        "gru_vs_xgb_all_original": etp.paired(match, y, J["gru"].to_numpy(), J["xgb_O"].to_numpy()),
        "xgb_all_extended_vs_original": etp.paired(match, y, J["xgb_E"].to_numpy(), J["xgb_O"].to_numpy()),
    }
    res = {"gru": a.tag, "rows": int(J.height), "matches": int(np.unique(match).size), "test_log_loss": ll, "pairs": pairs}
    os.makedirs(DIAG, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
