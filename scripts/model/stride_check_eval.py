"""Stride check: does sampling every 10th second change the exact-plant defuse-race results? (de_mirage 2026, frozen split, three seeds)

The main evals use every 10th second. This re-runs the two arms that carry the defuse-race verdict on EVERY second, with the same
57 inputs and the same fit settings:
  A      57 baseline inputs
  DX     + straight-line margin to the exact plant (defuse_race_de_mirage.parquet, every post-plant second)
  DWXc   + walking margin to the exact plant, centroid path (nav_plant_de_mirage_stride1.parquet, every post-plant second)
Each arm's stride-1 result is compared with the stride-10 result from defuse_race_eval.json (same arm, same seeds).

Pre-registered thresholds (written before any result was read):
  stride is adequate for an arm if |mean(stride-1 diff) - mean(stride-10 diff)| <= 0.0002 and the verdict (mean <= -0.0005 with every
  seed's CI upper bound < 0) is the same at both strides.
Caveat: this is one fit per arm and seed on about 4 million rows; the paired intervals at stride 1 are not directly comparable with the
stride-10 intervals because consecutive seconds are correlated.

Writes data/diagnostics/stride_check_eval.json.
Run in the cs2gpu environment (polars, xgboost):
  C:/Users/13144/.pixi/workspace-envs/cs2gpu-15282324412032665829/envs/default/python.exe scripts/model/stride_check_eval.py
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
OUT = os.path.join(DIAG, "stride_check_eval.json")
KEY = ["match_key", "round", "second"]
SEEDS = [0, 1, 2]
MISSING = D.MISSING


def fit_arm(G, cols, seed):
    tr = G.filter(pl.col("split") == "train")
    va = G.filter(pl.col("split") == "val")
    te = G.filter(pl.col("split") == "test")
    m, best = W.fit_xgb(tr.select(cols).to_numpy().astype(np.float32), tr["label_ct_win"].to_numpy().astype(np.int64),
                        va.select(cols).to_numpy().astype(np.float32), va["label_ct_win"].to_numpy().astype(np.int64),
                        mono=W.mono_for(cols), seed=seed, device="cuda")
    return te, W.predict(m, best, te.select(cols).to_numpy().astype(np.float32))


def main():
    t0 = time.time()
    feats = json.load(open(os.path.join(ROOT, "models", "winprob_v6_drop_heavy.meta.json"), encoding="utf-8"))["features"]
    assert len(feats) == 57, len(feats)
    D.STRIDE = 1
    G = D.base_rows(feats)
    dx = D.keyed(pl.read_parquet(os.path.join(DIAG, "defuse_race_de_mirage.parquet")).select(KEY + ["nearest_ct_dist"]))
    dx = dx.unique(KEY, keep="first", maintain_order=True)
    plant = D.keyed(pl.read_parquet(os.path.join(DIAG, "nav_plant_de_mirage_stride1.parquet"))).unique(KEY, keep="first", maintain_order=True)
    G = G.join(dx, on=KEY, how="left").join(plant, on=KEY, how="left")
    planted = (pl.col("bomb_planted") == 1) & pl.col("seconds_since_plant").is_not_null()
    fuse = D.FUSE - pl.col("seconds_since_plant")
    none = pl.lit(None, dtype=pl.Float64)
    fill = lambda c: pl.col(c).fill_null(MISSING).fill_nan(MISSING).clip(upper_bound=MISSING)  # noqa: E731
    G = G.with_columns(
        pl.when(planted).then(fuse - (fill("nearest_ct_dist") / D.SPEED + D.DEFUSE)).otherwise(none).alias("m_DX"),
        pl.when(planted).then(fuse - (fill("ct_path_plant") / D.SPEED + D.DEFUSE)).otherwise(none).alias("m_DWXc"),
    ).sort(KEY)
    arms = {"A": feats, "DX": feats + ["m_DX"], "DWXc": feats + ["m_DWXc"]}
    print(f"stride-1 rows {G.height:,}; test rows {G.filter(pl.col('split') == 'test').height:,} ({time.time() - t0:.0f} s)", flush=True)

    per_seed = {}
    te_ref = None
    for seed in SEEDS:
        preds = {}
        for name, cols in arms.items():
            te_k, p = fit_arm(G, cols, seed)
            preds[name] = p
            te_ref = te_k if te_ref is None else te_ref
        match = te_ref["match_key"].to_numpy()
        y = te_ref["label_ct_win"].to_numpy().astype(np.int64)
        per_seed[seed] = {"DX_minus_A": paired(match, y, preds["DX"], preds["A"]),
                          "DWXc_minus_A": paired(match, y, preds["DWXc"], preds["A"])}
        print(f"seed {seed}: DX-A {per_seed[seed]['DX_minus_A']['diff']:+.6f}  DWXc-A {per_seed[seed]['DWXc_minus_A']['diff']:+.6f} "
              f"({time.time() - t0:.0f} s)", flush=True)

    ref = json.load(open(os.path.join(DIAG, "defuse_race_eval2.json"), encoding="utf-8"))["per_seed"]
    ref_dx = {s: ref[str(s)]["DX_minus_A"]["diff"] for s in SEEDS}
    ref_dwx = {s: ref[str(s)]["DWX_minus_A"]["diff"] for s in SEEDS}
    comp = {}
    for name, refd in [("DX_minus_A", ref_dx), ("DWXc_minus_A", ref_dwx)]:
        d1 = [per_seed[s][name]["diff"] for s in SEEDS]
        ci1 = [per_seed[s][name]["ci95"] for s in SEEDS]
        d10 = [refd[s] for s in SEEDS]
        mean1, mean10 = float(np.mean(d1)), float(np.mean(d10))
        v1 = bool(mean1 <= -0.0005 and all(c[1] < 0 for c in ci1))
        v10 = bool(mean10 <= -0.0005)
        comp[name] = {"stride1_mean": mean1, "stride10_mean": mean10, "difference": mean1 - mean10,
                      "stride1_per_seed": d1, "stride1_per_seed_ci95": ci1,
                      "verdict_stride1_passes": v1, "stride10_mean_passes_mean_rule": v10,
                      "adequate": bool(abs(mean1 - mean10) <= 0.0002 and v1 == v10)}
    res = {"seeds": SEEDS, "per_seed": per_seed, "comparison_with_stride10": comp, "seconds": round(time.time() - t0)}
    os.makedirs(DIAG, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(comp, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
