"""Do walking distances over the navigation mesh add information to the 57-input XGBoost? (paired, de_mirage, frozen split)

Arms, all on the same rows (every 10th second of the frozen split; train rows fit, test rows score):
  A  57 baseline inputs (winprob_v6_drop_heavy list)
  P  A + 4 walking-distance features (nearest living CT and T walking distance to each bomb site, from build_nav_path_features.py)

Pre-registered thresholds (set before any result was read):
  walking distances add information if mean(P - A) <= -0.0005 and the 95% upper bound < 0.
  The test is the same rule as spatial_feature_eval.py, so the two results can be compared directly.

Writes data/diagnostics/nav_path_eval.json.
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
OUT = os.path.join(DIAG, "nav_path_eval.json")
KEY = ["match_key", "round", "second"]
STRIDE = 10
NAV_COLS = ["ct_path_siteA", "ct_path_siteB", "t_path_siteA", "t_path_siteB"]
MISSING = 1.0e4  # a side with no living player at that second: far away, not zero


def base_rows(feats):
    need = KEY + ["label_ct_win", "map_name"] + feats
    G = pl.read_parquet(os.path.join(ROOT, "data", "combined", "cache_snapshot_2026.parquet"), columns=need)
    G = W.drop_roster_defect_rounds(G)
    G = G.filter((pl.col("map_name") == "de_mirage") & pl.col("label_ct_win").is_not_null() & (pl.col("second") % STRIDE == 0))
    man = W.load_manifest(os.path.join(ROOT, "data", "combined", "split_manifest.json"))["assign"]
    A = pl.DataFrame({"match_key": list(man.keys()), "split": list(man.values())})
    return G.join(A, on="match_key", how="inner")


def fit(G, cols, name):
    tr = G.filter(pl.col("split") == "train")
    va = G.filter(pl.col("split") == "val")
    te = G.filter(pl.col("split") == "test")
    m, best = W.fit_xgb(tr.select(cols).to_numpy().astype(np.float32), tr["label_ct_win"].to_numpy().astype(np.int64),
                        va.select(cols).to_numpy().astype(np.float32), va["label_ct_win"].to_numpy().astype(np.int64),
                        mono=W.mono_for(cols), seed=0, device="cuda")
    p = W.predict(m, best, te.select(cols).to_numpy().astype(np.float32))
    print(f"  {name}: best iteration {best}, train {tr.height:,}, validation {va.height:,}, test {te.height:,}", flush=True)
    return te, p, best


def main():
    t0 = time.time()
    feats = json.load(open(os.path.join(ROOT, "models", "winprob_v6_drop_heavy.meta.json"), encoding="utf-8"))["features"]
    assert len(feats) == 57, len(feats)
    G = base_rows(feats)
    nav = pl.read_parquet(os.path.join(DIAG, "nav_path_de_mirage_stride10.parquet"))
    G = G.join(nav, on=KEY, how="left")
    print(f"base rows (stride {STRIDE}): {G.height:,}, with a walking distance for both sides: "
          f"{G.filter(pl.col('ct_path_siteA').is_not_null() & pl.col('t_path_siteA').is_not_null()).height:,}", flush=True)
    G = G.with_columns([pl.col(c).fill_null(MISSING).fill_nan(MISSING).clip(upper_bound=MISSING) for c in NAV_COLS]).sort(KEY)
    print("non-finite before fill:", {c: int(G.select(pl.col(c).is_infinite().sum()).item()) for c in NAV_COLS}, flush=True)
    arms = {"A": feats, "P": feats + NAV_COLS}
    preds, best, te = {}, {}, None
    for name, cols in arms.items():
        te_k, p, b = fit(G, cols, name)
        preds[name] = p
        best[name] = b
        te = te_k if te is None else te
    match = te["match_key"].to_numpy()
    y = te["label_ct_win"].to_numpy().astype(np.int64)
    r = paired(match, y, preds["P"], preds["A"])
    res = {
        "rows_test": int(te.height),
        "test_matches": int(np.unique(match).size),
        "best_iteration": best,
        "P_minus_A": r,
        "P_adds_information": bool(r["diff"] <= -0.0005 and r["ci95"][1] < 0),
        "thresholds": "walking distances add information if mean(P-A) <= -0.0005 and CI upper < 0",
        "seconds": round(time.time() - t0),
    }
    os.makedirs(DIAG, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
