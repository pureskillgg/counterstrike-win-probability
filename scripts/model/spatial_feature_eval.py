"""Do the spatial features help the 57-input XGBoost on de_mirage? (paired, GPU)

Arms, all trained and tested on the same de_mirage rows (the seconds that have a Voronoi value), in the same order:
  A  57 baseline inputs (winprob_v6_drop_heavy list)                         <- the reference for this test
  V  A + Voronoi proximity CT control (voronoi_control_de_mirage.parquet)
  D  A + defuse-race margin v1 (defuse_race_de_mirage.parquet; NaN outside post-plant)
  VD A + both

Pre-registered thresholds (set before any result was read):
  a feature adds information if mean(arm - A) <= -0.0005 and the 95% upper bound < 0.
  The defuse margin is computed from straight-line distance (v1), so a null result does not rule out the navigation-mesh version.

Single map (de_mirage), so these numbers are about the feature, not the cross-map comparison.

  ...cs2gpu python.exe -I -X utf8 -u scripts\\model\\spatial_feature_eval.py
Writes data/diagnostics/spatial_feature_eval.json.
"""
import json
import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
import wp_common as W  # noqa: E402

ROOT = W.ROOT
DIAG = os.path.join(ROOT, "data", "diagnostics")
OUT = os.path.join(DIAG, "spatial_feature_eval.json")
KEY = ["match_key", "round", "second"]
N_BOOT = 1000
STRIDE = 2


def base_rows():
    feats = json.load(open(os.path.join(ROOT, "models", "winprob_v6_drop_heavy.meta.json"), encoding="utf-8"))["features"]
    assert len(feats) == 57, len(feats)
    need = KEY + ["label_ct_win", "map_name"] + feats
    G = pl.read_parquet(os.path.join(ROOT, "data", "combined", "cache_snapshot_2026.parquet"), columns=need)
    G = W.drop_roster_defect_rounds(G)
    G = G.filter((pl.col("map_name") == "de_mirage") & pl.col("label_ct_win").is_not_null())
    man = W.load_manifest(os.path.join(ROOT, "data", "combined", "split_manifest.json"))["assign"]
    A = pl.DataFrame({"match_key": list(man.keys()), "split": list(man.values())})
    return G.join(A, on="match_key", how="inner"), feats


def paired(match, y, pa, pb, seed=0):
    la = W.ll_vec(pa, y)
    lb = W.ll_vec(pb, y)
    uniq, inv = np.unique(match, return_inverse=True)
    M = len(uniq)
    cnt = np.bincount(inv, minlength=M).astype(float)
    sa = np.bincount(inv, weights=la, minlength=M)
    sb = np.bincount(inv, weights=lb, minlength=M)
    rng = np.random.default_rng(seed)
    d = np.empty(N_BOOT)
    for b in range(N_BOOT):
        w = np.bincount(rng.integers(0, M, M), minlength=M).astype(float)
        d[b] = ((w * sa).sum() - (w * sb).sum()) / (w * cnt).sum()
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"diff": float(la.mean() - lb.mean()), "ci95": [float(lo), float(hi)], "rows": int(len(y)), "matches": int(M)}


def fit(G, cols, name):
    tr = G.filter((pl.col("split") == "train") & (pl.col("second") % STRIDE == 0))
    va = G.filter(pl.col("split") == "val")
    te = G.filter(pl.col("split") == "test")
    m, best = W.fit_xgb(tr.select(cols).to_numpy().astype(np.float32), tr["label_ct_win"].to_numpy().astype(np.int64),
                        va.select(cols).to_numpy().astype(np.float32), va["label_ct_win"].to_numpy().astype(np.int64),
                        mono=W.mono_for(cols), seed=0, device="cuda")
    p = W.predict(m, best, te.select(cols).to_numpy().astype(np.float32))
    print(f"  {name}: best iteration {best}, test rows {te.height:,}", flush=True)
    return te, p, best


def main():
    t0 = time.time()
    G, feats = base_rows()
    vor = pl.read_parquet(os.path.join(DIAG, "voronoi_control_de_mirage.parquet"), columns=KEY + ["ct_voronoi"])
    dm = pl.read_parquet(os.path.join(DIAG, "defuse_race_de_mirage.parquet"), columns=KEY + ["defuse_margin"])
    G = G.join(vor, on=KEY, how="inner").join(dm, on=KEY, how="left")
    G = G.filter(pl.col("ct_voronoi").is_not_null()).sort(KEY)
    print(f"common rows: {G.height:,} seconds, {G['match_key'].n_unique()} matches", flush=True)
    arms = {"A": feats, "V": feats + ["ct_voronoi"], "D": feats + ["defuse_margin"], "VD": feats + ["ct_voronoi", "defuse_margin"]}
    preds, best, te = {}, {}, None
    for name, cols in arms.items():
        te_k, p, b = fit(G, cols, name)
        preds[name] = p
        best[name] = b
        te = te_k if te is None else te
    match = te["match_key"].to_numpy()
    y = te["label_ct_win"].to_numpy().astype(np.int64)
    res = {"rows_test": int(te.height), "best_iteration": best}
    for name in ["V", "D", "VD"]:
        r = paired(match, y, preds[name], preds["A"])
        res[f"{name}_minus_A"] = r
        res[f"{name}_adds_information"] = bool(r["diff"] <= -0.0005 and r["ci95"][1] < 0)
    res["thresholds"] = "adds information if mean(arm - A) <= -0.0005 and CI upper < 0"
    res["seconds"] = round(time.time() - t0)
    os.makedirs(DIAG, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
