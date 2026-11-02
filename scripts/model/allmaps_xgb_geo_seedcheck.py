"""Seed check for the all-maps distance XGBoost: the same three arms as allmaps_xgb_geo.py, trained with seeds 1 and 2.

allmaps_xgb_geo.py (seed 0) is the third seed; its result file is read here only to form the three-seed mean. Arms, data, split, recipe and test set are identical to allmaps_xgb_geo.py: frozen recipe (W.fit_xgb: depth 4, eta 0.05, mcw 20, subsample 0.8, early stopping on the all-maps validation split), test = the de_mirage test matches (865,974 seconds, 687 matches), paired by spatial_feature_eval.paired (1,000 resamples, seed 0).

Pre-registered (written before any seed-1 or seed-2 result): the distance features are confirmed if
  (a) for EACH seed in {1, 2}, S minus A and W minus A each have a mean <= -0.0005 and a 95% CI upper bound < 0; and
  (b) over seeds {0, 1, 2}, the mean of S minus A and the mean of W minus A are each <= -0.0005.
Otherwise the single-seed result is not confirmed.

Run in the cs2gpu environment (polars, xgboost, CUDA):
  C:/Users/<user>/.pixi/workspace-envs/cs2gpu-15282324412032665829/envs/default/python.exe scripts/model/allmaps_xgb_geo_seedcheck.py
Writes data/diagnostics/allmaps_xgb_geo_seedcheck.json.
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

ROOT = W.ROOT
DIAG = os.path.join(ROOT, "data", "diagnostics")
OUT = os.path.join(DIAG, "allmaps_xgb_geo_seedcheck.json")
SEED0 = os.path.join(DIAG, "allmaps_xgb_geo.json")
MAPS = ["de_mirage", "de_dust2", "de_ancient", "de_anubis", "de_inferno", "de_nuke", "de_cache"]
KEY = ["match_key", "round", "second"]
ST = [f"st{i}" for i in range(4)]
WK = [f"wk{i}" for i in range(4)]
SEEDS = [1, 2]


def main():
    t0 = time.time()
    feats = json.load(open(os.path.join(ROOT, "models", "winprob_v6_drop_heavy.meta.json"), encoding="utf-8"))["features"]
    assert len(feats) == 57, len(feats)
    geo = pl.concat([pl.read_parquet(os.path.join(DIAG, "allmaps_geo", f"{m}.parquet")) for m in MAPS])
    C = pl.read_parquet(os.path.join(ROOT, "data", "combined", "cache_snapshot_2026_ext.parquet"),
                        columns=sorted(set(KEY + ["match_key", "label_ct_win", "map_name"] + feats)))
    C = W.drop_roster_defect_rounds(C)
    C = C.filter(pl.col("label_ct_win").is_not_null())
    C = C.with_columns(pl.col("round").cast(pl.Int64), pl.col("second").cast(pl.Int64))
    man = W.load_manifest(os.path.join(ROOT, "data", "combined", "split_manifest.json"))["assign"]
    A = pl.DataFrame({"match_key": list(man.keys()), "split": list(man.values())})
    G = C.join(geo, on=KEY, how="inner").join(A, on="match_key", how="inner")
    G = G.unique(KEY, keep="first", maintain_order=True).sort(KEY)
    print(f"joined rows {G.height:,} ({time.time() - t0:.0f} s)", flush=True)
    tr = G.filter(pl.col("split") == "train")
    va = G.filter(pl.col("split") == "val")
    te = G.filter((pl.col("split") == "test") & (pl.col("map_name") == "de_mirage"))
    y_te = te["label_ct_win"].to_numpy().astype(np.int64)
    match_te = te["match_key"].to_numpy()
    arms = {"A": feats, "S": feats + ST, "W": feats + WK}
    per_seed = []
    for seed in SEEDS:
        preds = {}
        for name, cols in arms.items():
            m, best = W.fit_xgb(tr.select(cols).to_numpy().astype(np.float32), tr["label_ct_win"].to_numpy().astype(np.int64),
                                va.select(cols).to_numpy().astype(np.float32), va["label_ct_win"].to_numpy().astype(np.int64),
                                mono=W.mono_for(cols), seed=seed, device="cuda")
            preds[name] = W.predict(m, best, te.select(cols).to_numpy().astype(np.float32))
            print(f"seed {seed} arm {name}: test log loss {float(W.ll_vec(preds[name], y_te).mean()):.6f}, best iteration {best} "
                  f"({time.time() - t0:.0f} s)", flush=True)
        row = {"seed": seed,
               "log_loss": {n: float(W.ll_vec(p, y_te).mean()) for n, p in preds.items()},
               "S_minus_A": paired(match_te, y_te, preds["S"], preds["A"]),
               "W_minus_A": paired(match_te, y_te, preds["W"], preds["A"])}
        for k in ["S", "W"]:
            d = row[f"{k}_minus_A"]
            row[f"{k}_passes"] = bool(d["diff"] <= -0.0005 and d["ci95"][1] < 0)
        per_seed.append(row)
        print(json.dumps({k: row[k] for k in ["seed", "S_passes", "W_passes"]}), flush=True)
    s0 = json.load(open(SEED0, encoding="utf-8"))
    seed0 = {"seed": 0, "log_loss": s0["log_loss"], "S_minus_A": s0["S_minus_A"], "W_minus_A": s0["W_minus_A"],
             "S_passes": s0["S_passes"], "W_passes": s0["W_passes"]}
    rows = [seed0] + per_seed
    means = {k: float(np.mean([r[f"{k}_minus_A"]["diff"] for r in rows])) for k in ["S", "W"]}
    res = {"seeds": rows, "mean_S_minus_A": means["S"], "mean_W_minus_A": means["W"],
           "confirmed_S": bool(all(r["S_passes"] for r in per_seed) and means["S"] <= -0.0005),
           "confirmed_W": bool(all(r["W_passes"] for r in per_seed) and means["W"] <= -0.0005),
           "seconds": round(time.time() - t0)}
    os.makedirs(DIAG, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ["mean_S_minus_A", "mean_W_minus_A", "confirmed_S", "confirmed_W"]}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
