"""All-maps XGBoost with distance features, matched in scope to the all-maps GRU runs.

Arms (frozen recipe, W.fit_xgb: depth 4, eta 0.05, mcw 20, subsample 0.8, early stopping on the all-maps validation split):
  A   the 57 frozen inputs (reference; should reproduce the all-maps extended XGBoost, 0.4507 on the de_mirage test)
  S   A + straight-line distances (st0..st3 from allmaps_geo_rows.py)
  W   A + walking distances (wk0..wk3)
Train: the train split of all seven maps. Validation: the validation split of all maps. Test: the frozen de_mirage test matches
(865,974 seconds, 687 matches), the same rows as the GRU comparisons.

Pre-registered (written before any result was read): a distance arm beats A if mean paired log-loss difference (arm minus A) over the
test seconds is <= -0.0005 and its 95% match-cluster CI upper bound is < 0 (spatial_feature_eval.paired, 1,000 resamples). One seed.

Run in the cs2gpu environment (polars, xgboost, CUDA):
  C:/Users/13144/.pixi/workspace-envs/cs2gpu-15282324412032665829/envs/default/python.exe scripts/model/allmaps_xgb_geo.py
Writes data/diagnostics/allmaps_xgb_geo.json.
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
OUT = os.path.join(DIAG, "allmaps_xgb_geo.json")
MAPS = ["de_mirage", "de_dust2", "de_ancient", "de_anubis", "de_inferno", "de_nuke", "de_cache"]
KEY = ["match_key", "round", "second"]
ST = [f"st{i}" for i in range(4)]
WK = [f"wk{i}" for i in range(4)]


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
    preds = {}
    for name, cols in arms.items():
        m, best = W.fit_xgb(tr.select(cols).to_numpy().astype(np.float32), tr["label_ct_win"].to_numpy().astype(np.int64),
                            va.select(cols).to_numpy().astype(np.float32), va["label_ct_win"].to_numpy().astype(np.int64),
                            mono=W.mono_for(cols), seed=0, device="cuda")
        preds[name] = W.predict(m, best, te.select(cols).to_numpy().astype(np.float32))
        print(f"arm {name}: test log loss {float(W.ll_vec(preds[name], y_te).mean()):.6f}, best iteration {best} "
              f"({time.time() - t0:.0f} s)", flush=True)
    res = {"rows_train": tr.height, "rows_val": va.height, "rows_test": te.height,
           "test_matches": int(te["match_key"].n_unique()),
           "log_loss": {n: float(W.ll_vec(p, y_te).mean()) for n, p in preds.items()},
           "S_minus_A": paired(match_te, y_te, preds["S"], preds["A"]),
           "W_minus_A": paired(match_te, y_te, preds["W"], preds["A"]),
           "S_passes": None, "W_passes": None, "seconds": round(time.time() - t0)}
    for k in ["S", "W"]:
        d = res[f"{k}_minus_A"]
        res[f"{k}_passes"] = bool(d["diff"] <= -0.0005 and d["ci95"][1] < 0)
    os.makedirs(DIAG, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ["log_loss", "S_passes", "W_passes"]}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
