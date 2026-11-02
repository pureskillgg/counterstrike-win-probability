"""G1: does the all-maps GRU beat the 57-input XGBoost on validation, per seed? (pre-registered, Meridian pin 62e3f1d6).

Rule (standard): for EACH seed, mean(GRU minus XGB-A) on validation <= -0.0005 with a 95% match-clustered CI upper bound < 0,
and the three-seed mean <= -0.0005. Test seconds are descriptive only (not written here).
Writes data/diagnostics/g1_gru_vs_xgb_validation.json.
"""
import json
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spatial_feature_eval import paired  # noqa: E402

ROOT = "G:/pureskill/csds-cs2"
DIAG = os.path.join(ROOT, "data", "diagnostics")
TEMP = os.path.join(ROOT, "data", "temporal")
OUT = os.path.join(DIAG, "g1_gru_vs_xgb_validation.json")
KEY = ["match_key", "round", "second"]
SEEDS = [0, 1, 2]


def gru_tag(seed):
    return "allmaps_gru_loadout" + ("" if seed == 0 else f"_seed{seed}") + "_valpreds"


def main():
    per_seed = {}
    for seed in SEEDS:
        g = pl.read_parquet(os.path.join(TEMP, f"{gru_tag(seed)}_val_preds.parquet")).select(KEY + ["label", "p"]).rename({"p": "p_gru"})
        x = pl.read_parquet(os.path.join(DIAG, "preds", f"allmaps_xgb_A_seed{seed}.parquet"))
        x = x.filter(pl.col("split") == "val").select(KEY + ["p"]).rename({"p": "p_xgb"})
        j = g.join(x, on=KEY, how="inner")
        y = j["label"].to_numpy().astype(np.int64)
        m = j["match_key"].to_numpy()
        d = paired(m, y, j["p_gru"].to_numpy(), j["p_xgb"].to_numpy())
        d["aligned_rows"] = j.height
        d["gru_rows"] = g.height
        d["xgb_rows"] = x.height
        per_seed[seed] = {"GRU_minus_XGBA_validation": d}
        print(f"seed {seed}: GRU minus XGB-A validation {d['diff']:+.6f} CI [{d['ci95'][0]:+.6f}, {d['ci95'][1]:+.6f}] "
              f"aligned {j.height:,} (GRU {g.height:,}, XGB {x.height:,})", flush=True)
    vals = [per_seed[s]["GRU_minus_XGBA_validation"]["diff"] for s in SEEDS]
    cis = [per_seed[s]["GRU_minus_XGBA_validation"]["ci95"] for s in SEEDS]
    mean = float(np.mean(vals))
    passes = bool(mean <= -0.0005 and all(c[1] < 0 for c in cis))
    res = {"per_seed": per_seed, "mean_over_seeds": mean, "passes": passes,
           "rule": "for each seed mean <= -0.0005 with CI upper < 0, and three-seed mean <= -0.0005 (validation)"}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({"mean_over_seeds": mean, "passes": passes}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
