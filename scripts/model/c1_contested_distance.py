"""C1: does the distance gain (SN minus A) survive on contested (even) rounds? Validation seconds, saved predictions, no refit.

Pre-registration (Meridian pin a2b2c8a0): contested = alive_ct == alive_t and |st_equip_ct - st_equip_t| <= 1500 (Huang et al. 2026).
Rule (decision, validation contested seconds): for EACH seed, mean(SN minus A) <= -0.0005 with 95% match-clustered CI upper bound < 0,
and the three-seed mean <= -0.0005. Uncontested seconds and the test seconds are reported descriptively only.

Writes data/diagnostics/c1_contested_distance.json.
Run in the cs2gpu environment (polars):
  C:/Users/<user>/.pixi/workspace-envs/cs2gpu-15282324412032665829/envs/default/python.exe -I -X utf8 -u scripts/model/c1_contested_distance.py
"""
import json
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
from spatial_feature_eval import paired  # noqa: E402

ROOT = "G:/pureskill/csds-cs2"
DIAG = os.path.join(ROOT, "data", "diagnostics")
OUT = os.path.join(DIAG, "c1_contested_distance.json")
KEY = ["match_key", "round", "second"]
SEEDS = [0, 1, 2]


def load(seed, arm):
    p = pl.read_parquet(os.path.join(DIAG, "preds", f"allmaps_xgb_{arm}_seed{seed}.parquet"))
    return p.filter(pl.col("split") == "val").with_columns(pl.col("round").cast(pl.Int64), pl.col("second").cast(pl.Int64))


def main():
    cache = pl.read_parquet(os.path.join(ROOT, "data", "combined", "cache_snapshot_2026_ext.parquet"),
                            columns=KEY + ["alive_ct", "alive_t", "st_equip_ct", "st_equip_t"])
    cache = cache.with_columns(pl.col("round").cast(pl.Int64), pl.col("second").cast(pl.Int64)).unique(KEY, keep="first")
    per_seed = {}
    contested_share = None
    for seed in SEEDS:
        a = load(seed, "A")
        s = load(seed, "SN")
        j = a.join(s.select(KEY + ["p"]).rename({"p": "p_SN"}), on=KEY, how="inner").rename({"p": "p_A"})
        j = j.join(cache.select(KEY + ["alive_ct", "alive_t", "st_equip_ct", "st_equip_t"]), on=KEY, how="left")
        contested = ((pl.col("alive_ct") == pl.col("alive_t")) &
                     ((pl.col("st_equip_ct") - pl.col("st_equip_t")).abs() <= 1500))
        j = j.with_columns(contested.alias("contested")).filter(pl.col("alive_ct").is_not_null())
        if contested_share is None:
            contested_share = float(j["contested"].mean())
        y = j["label"].to_numpy().astype(np.int64)
        match = j["match_key"].to_numpy()
        c = j["contested"].to_numpy()
        res = {}
        for name, mask in [("contested", c), ("uncontested", ~c)]:
            sub_match = match[mask]
            res[name] = paired(sub_match, y[mask], j["p_SN"].to_numpy()[mask], j["p_A"].to_numpy()[mask])
            res[name]["rows"] = int(mask.sum())
        per_seed[seed] = res
        print(f"seed {seed}: contested SN-A {res['contested']['diff']:+.6f} "
              f"CI [{res['contested']['ci95'][0]:+.6f}, {res['contested']['ci95'][1]:+.6f}]; "
              f"uncontested {res['uncontested']['diff']:+.6f} ({res['contested']['rows']:,} contested rows)", flush=True)

    d_c = [per_seed[s]["contested"]["diff"] for s in SEEDS]
    ci_c = [per_seed[s]["contested"]["ci95"] for s in SEEDS]
    mean = float(np.mean(d_c))
    passes = bool(mean <= -0.0005 and all(c_[1] < 0 for c_ in ci_c))
    res = {"contested_share_validation": contested_share, "per_seed": per_seed,
           "mean_contested_SN_minus_A": mean, "passes_contested_rule": passes,
           "rule": "for each seed mean <= -0.0005 with CI upper < 0, and three-seed mean <= -0.0005 (validation contested seconds)"}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({"contested_share": contested_share, "mean_contested": mean, "passes": passes}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
