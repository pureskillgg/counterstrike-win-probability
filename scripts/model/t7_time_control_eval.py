"""T7: does travel-time map control add to the defuse-margin XGBoost? (de_mirage, every 10th second, three seeds, pre-registered)

Pre-registration (Meridian pin a1f143d9, written before any result was read):
  Arms (validation seconds, decision set): DX = 57 inputs plus the exact-plant straight-line defuse margin (as in defuse_race_eval.py);
  DXT = DX plus travel-time control features t7_ct, t7_t, t7_net (map_control_time.py).
  Rule (standard): pass if for EACH seed mean(DXT minus DX) on validation <= -0.0005 with a 95% match-clustered CI upper bound < 0,
  and the three-seed mean <= -0.0005. Test seconds are scored and saved for the record only, and do not enter the decision.
  Descriptive only: DXT minus the distance-based control already tested (not re-run here).

Rows: the same base rows as map_control_eval.py. Snapshots with no travel-time control stay null (XGBoost handles missing values);
the number of such rows is reported. Exclusion counts for players without a navigation area are in map_control_time's log.

Writes data/diagnostics/t7_time_control.json.
Run in the cs2gpu environment (polars, xgboost, CUDA):
  C:/Users/<user>/.pixi/workspace-envs/cs2gpu-15282324412032665829/envs/default/python.exe -I -X utf8 -u scripts/model/t7_time_control_eval.py
"""
import json
import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
import wp_common as W  # noqa: E402
from spatial_feature_eval import paired  # noqa: E402
import defuse_race_eval as D  # noqa: E402

ROOT = W.ROOT
DIAG = os.path.join(ROOT, "data", "diagnostics")
OUT = os.path.join(DIAG, "t7_time_control.json")
KEY = ["match_key", "round", "second"]
SEEDS = [0, 1, 2]
T7 = ["t7_ct", "t7_t", "t7_net"]
MISSING = D.MISSING


def main():
    t0 = time.time()
    feats = json.load(open(os.path.join(ROOT, "models", "winprob_v6_drop_heavy.meta.json"), encoding="utf-8"))["features"]
    assert len(feats) == 57, len(feats)
    G = D.base_rows(feats)
    t7 = D.keyed(pl.read_parquet(os.path.join(DIAG, "map_control_time_de_mirage_stride10.parquet")))
    t7 = t7.unique(KEY, keep="first", maintain_order=True)
    dx = D.keyed(pl.read_parquet(os.path.join(DIAG, "defuse_race_de_mirage.parquet")).select(KEY + ["nearest_ct_dist"]))
    dx = dx.unique(KEY, keep="first", maintain_order=True)
    G = (G.join(t7, on=KEY, how="left").join(dx, on=KEY, how="left")
          .join(D.plant_sites().select(["match_key", "round", "site"]), on=["match_key", "round"], how="left"))
    planted = (pl.col("bomb_planted") == 1) & pl.col("seconds_since_plant").is_not_null() & pl.col("site").is_not_null()
    fill = pl.col("nearest_ct_dist").fill_null(MISSING).fill_nan(MISSING).clip(upper_bound=MISSING)
    G = G.with_columns(
        pl.when(planted).then(D.FUSE - pl.col("seconds_since_plant") - (fill / D.SPEED + D.DEFUSE)).otherwise(None).alias("m_DX")
    ).sort(KEY)
    n_na = G.filter(pl.col("t7_net").is_null()).height
    print(f"rows {G.height:,}; snapshots without travel-time control {n_na:,} ({time.time() - t0:.0f} s)", flush=True)

    arms = {"DX": feats + ["m_DX"], "DXT": feats + ["m_DX"] + T7}
    tr = G.filter(pl.col("split") == "train")
    va = G.filter(pl.col("split") == "val")
    te = G.filter(pl.col("split") == "test")
    y_va = va["label_ct_win"].to_numpy().astype(np.int64)
    y_te = te["label_ct_win"].to_numpy().astype(np.int64)
    match_va = va["match_key"].to_numpy()
    match_te = te["match_key"].to_numpy()

    per_seed = {}
    for seed in SEEDS:
        preds_va, preds_te = {}, {}
        for name, cols in arms.items():
            m, best = W.fit_xgb(tr.select(cols).to_numpy().astype(np.float32), tr["label_ct_win"].to_numpy().astype(np.int64),
                                va.select(cols).to_numpy().astype(np.float32), y_va,
                                mono=W.mono_for(cols), seed=seed, device="cuda")
            preds_va[name] = W.predict(m, best, va.select(cols).to_numpy().astype(np.float32))
            preds_te[name] = W.predict(m, best, te.select(cols).to_numpy().astype(np.float32))
            print(f"seed {seed} {name}: validation log loss {float(W.ll_vec(preds_va[name], y_va).mean()):.6f}, "
                  f"best iteration {best} ({time.time() - t0:.0f} s)", flush=True)
        per_seed[seed] = {
            "DXT_minus_DX_validation": paired(match_va, y_va, preds_va["DXT"], preds_va["DX"]),
            "DXT_minus_DX_test_descriptive": paired(match_te, y_te, preds_te["DXT"], preds_te["DX"]),
        }
        d = per_seed[seed]["DXT_minus_DX_validation"]
        print(f"seed {seed}: validation DXT minus DX {d['diff']:+.6f} CI [{d['ci95'][0]:+.6f}, {d['ci95'][1]:+.6f}] "
              f"({time.time() - t0:.0f} s)", flush=True)

    d_va = [per_seed[s]["DXT_minus_DX_validation"]["diff"] for s in SEEDS]
    ci_va = [per_seed[s]["DXT_minus_DX_validation"]["ci95"] for s in SEEDS]
    mean = float(np.mean(d_va))
    passes = bool(mean <= -0.0005 and all(c[1] < 0 for c in ci_va))
    res = {"decision_set": "validation seconds", "rows_without_travel_time_control": int(n_na), "seeds": SEEDS,
           "per_seed": per_seed, "mean_over_seeds_validation": mean, "passes": passes,
           "rule": "mean(DXT minus DX) <= -0.0005 and every seed's 95% CI upper bound < 0 (validation)",
           "seconds": round(time.time() - t0)}
    os.makedirs(DIAG, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({"mean_over_seeds_validation": mean, "passes": passes}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
