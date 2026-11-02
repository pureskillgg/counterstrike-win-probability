"""B1: does a fixed 50/50 blend of the GRU and the distance XGBoost (SN) beat either alone on validation? (pre-registered, pin
f5eec3a3-era B1 entry 3941788a). No weight is fitted. Rule per comparison (blend minus SN; blend minus GRU): for EACH seed mean <= -0.0005
with CI upper < 0, and the three-seed mean <= -0.0005. Test is not used. Writes data/diagnostics/b1_blend_validation.json.
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
OUT = os.path.join(DIAG, "b1_blend_validation.json")
KEY = ["match_key", "round", "second"]
SEEDS = [0, 1, 2]


def main():
    per = {}
    for seed in SEEDS:
        tag = "allmaps_gru_loadout" + ("" if seed == 0 else f"_seed{seed}") + "_valpreds"
        g = pl.read_parquet(os.path.join(TEMP, f"{tag}_val_preds.parquet")).select(KEY + ["label", "p"]).rename({"p": "p_gru"})
        s = pl.read_parquet(os.path.join(DIAG, "preds", f"allmaps_xgb_SN_seed{seed}.parquet"))
        s = s.filter(pl.col("split") == "val").select(KEY + ["p"]).rename({"p": "p_sn"})
        j = g.join(s, on=KEY, how="inner").with_columns(((pl.col("p_gru") + pl.col("p_sn")) / 2).alias("p_blend"))
        y = j["label"].to_numpy().astype(np.int64)
        m = j["match_key"].to_numpy()
        b = j["p_blend"].to_numpy()
        per[seed] = {
            "blend_minus_SN": paired(m, y, b, j["p_sn"].to_numpy()),
            "blend_minus_GRU": paired(m, y, b, j["p_gru"].to_numpy()),
            "aligned_rows": j.height,
        }
        print(f"seed {seed}: blend-SN {per[seed]['blend_minus_SN']['diff']:+.6f} "
              f"{per[seed]['blend_minus_SN']['ci95']} | blend-GRU {per[seed]['blend_minus_GRU']['diff']:+.6f} "
              f"{per[seed]['blend_minus_GRU']['ci95']} | aligned {j.height:,}", flush=True)
    verdict = {}
    for name in ["blend_minus_SN", "blend_minus_GRU"]:
        d = [per[s][name]["diff"] for s in SEEDS]
        ci = [per[s][name]["ci95"] for s in SEEDS]
        mean = float(np.mean(d))
        verdict[name] = {"mean": mean, "passes": bool(mean <= -0.0005 and all(c[1] < 0 for c in ci))}
    res = {"per_seed": per, "verdicts": verdict, "fixed_weight": 0.5}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(verdict, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
