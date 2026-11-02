"""Walking versus straight-line distance to the bomb, on the 57-input XGBoost (paired, de_mirage, frozen split, three seeds).

Arms, all on the same rows (every 10th second of the frozen split):
  A   57 baseline inputs (the baseline has no distance to the bomb; only bomb_planted)
  S   A + straight-line distance from the nearest living CT and T to each site (4 features)
  P   A + walking distance over the navigation mesh, same four quantities (4 features)
  SP  A + both sets (8 features)

Pre-registered thresholds (written before any result was read):
  P adds information   if mean(P - A) <= -0.0005 and the 95% upper bound < 0
  S adds information   if mean(S - A) <= -0.0005 and the 95% upper bound < 0
  walking beats straight-line if mean(P - S) <= -0.0005 and the 95% upper bound < 0
  (the bootstrap is the match-cluster bootstrap in spatial_feature_eval.paired, 1000 resamples)
Each seed is a separate fit of every arm; the seed-to-seed spread is reported with each difference.

Writes data/diagnostics/nav_path_eval2.json.
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
OUT = os.path.join(DIAG, "nav_path_eval2.json")
KEY = ["match_key", "round", "second"]
STRIDE = 10
SEEDS = [0, 1, 2]
PATH = ["ct_path_siteA", "ct_path_siteB", "t_path_siteA", "t_path_siteB"]
STRAIGHT = ["st_ct_siteA", "st_ct_siteB", "st_t_siteA", "st_t_siteB"]
MISSING = 1.0e4  # unreachable (infinite) or no living player on that side: far away, not zero


def base_rows(feats):
    need = KEY + ["label_ct_win", "map_name"] + feats
    G = pl.read_parquet(os.path.join(ROOT, "data", "combined", "cache_snapshot_2026.parquet"), columns=need)
    G = W.drop_roster_defect_rounds(G)
    G = G.filter((pl.col("map_name") == "de_mirage") & pl.col("label_ct_win").is_not_null() & (pl.col("second") % STRIDE == 0))
    man = W.load_manifest(os.path.join(ROOT, "data", "combined", "split_manifest.json"))["assign"]
    A = pl.DataFrame({"match_key": list(man.keys()), "split": list(man.values())})
    return G.join(A, on="match_key", how="inner")


def fit(G, cols, seed):
    tr = G.filter(pl.col("split") == "train")
    va = G.filter(pl.col("split") == "val")
    te = G.filter(pl.col("split") == "test")
    m, best = W.fit_xgb(tr.select(cols).to_numpy().astype(np.float32), tr["label_ct_win"].to_numpy().astype(np.int64),
                        va.select(cols).to_numpy().astype(np.float32), va["label_ct_win"].to_numpy().astype(np.int64),
                        mono=W.mono_for(cols), seed=seed, device="cuda")
    return te, W.predict(m, best, te.select(cols).to_numpy().astype(np.float32)), best


def main():
    t0 = time.time()
    feats = json.load(open(os.path.join(ROOT, "models", "winprob_v6_drop_heavy.meta.json"), encoding="utf-8"))["features"]
    assert len(feats) == 57, len(feats)
    G = base_rows(feats)
    nav = pl.read_parquet(os.path.join(DIAG, "nav_path_de_mirage_stride10.parquet"))
    G = G.join(nav, on=KEY, how="left")
    fill = [pl.col(c).fill_null(MISSING).fill_nan(MISSING).clip(upper_bound=MISSING) for c in PATH + STRAIGHT]
    G = G.with_columns(fill).sort(KEY)
    print(f"rows (stride {STRIDE}): {G.height:,}", flush=True)
    arms = {"A": feats, "S": feats + STRAIGHT, "P": feats + PATH, "SP": feats + STRAIGHT + PATH}
    per_seed = {}
    te_ref = None
    for seed in SEEDS:
        preds, best = {}, {}
        for name, cols in arms.items():
            te_k, p, b = fit(G, cols, seed)
            preds[name], best[name] = p, b
            te_ref = te_k if te_ref is None else te_ref
        match = te_ref["match_key"].to_numpy()
        y = te_ref["label_ct_win"].to_numpy().astype(np.int64)
        per_seed[seed] = {
            "best_iteration": best,
            "P_minus_A": paired(match, y, preds["P"], preds["A"]),
            "S_minus_A": paired(match, y, preds["S"], preds["A"]),
            "P_minus_S": paired(match, y, preds["P"], preds["S"]),
            "SP_minus_P": paired(match, y, preds["SP"], preds["P"]),
        }
        print(f"seed {seed}: P-A {per_seed[seed]['P_minus_A']['diff']:+.6f}, S-A {per_seed[seed]['S_minus_A']['diff']:+.6f}, "
              f"P-S {per_seed[seed]['P_minus_S']['diff']:+.6f}  ({time.time() - t0:.0f} s)", flush=True)

    def rule(name):
        d = [per_seed[s][name]["diff"] for s in SEEDS]
        ci = [per_seed[s][name]["ci95"] for s in SEEDS]
        mean = float(np.mean(d))
        return {"mean_over_seeds": mean, "per_seed": d, "per_seed_ci95": ci,
                "passes": bool(mean <= -0.0005 and all(c[1] < 0 for c in ci))}

    res = {"rows_test": int(te_ref.height), "test_matches": int(np.unique(te_ref["match_key"].to_numpy()).size),
           "seeds": SEEDS, "per_seed": per_seed,
           "verdicts": {"P_adds_information": rule("P_minus_A"), "S_adds_information": rule("S_minus_A"),
                        "walking_beats_straight": rule("P_minus_S")},
           "thresholds": "mean(diff) <= -0.0005 and every seed's CI upper bound < 0",
           "seconds": round(time.time() - t0)}
    os.makedirs(DIAG, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res["verdicts"], indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
