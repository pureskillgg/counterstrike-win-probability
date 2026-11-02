"""Does walking-travel map control add information to the 57-input XGBoost, and does it add beyond the defuse-race margin?
(de_mirage 2026, frozen split, every 10th second, three seeds, paired match-cluster bootstrap)

Map control per snapshot from map_control_features.py (awpy reachability method): mc_ct, mc_t, mc_contested (size-weighted fractions).

Arms (same rows):
  A       57 baseline inputs
  MC      A + mc_ct, mc_t, mc_contested
  DX      A + straight-line margin to the exact plant (10 s defuse, 250 units/s), as in defuse_race_eval.py
  DXMC    DX + mc_ct, mc_t, mc_contested

Pre-registered thresholds (written before any result was read):
  map control adds information     mean(MC - A) <= -0.0005 and every seed's 95% CI upper bound < 0
  map control adds beyond DX       mean(DXMC - DX) <= -0.0005 and every seed's 95% CI upper bound < 0
Match-cluster bootstrap: spatial_feature_eval.paired (1,000 resamples, seed 0). Snapshots awpy could not evaluate stay null (NaN).

Writes data/diagnostics/map_control_eval.json.
Run in the cs2gpu environment (polars, xgboost):
  C:/Users/13144/.pixi/workspace-envs/cs2gpu-15282324412032665829/envs/default/python.exe scripts/model/map_control_eval.py
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
OUT = os.path.join(DIAG, "map_control_eval.json")
KEY = ["match_key", "round", "second"]
SEEDS = [0, 1, 2]
MC = ["mc_ct", "mc_t", "mc_contested"]
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
    G = D.base_rows(feats)
    mc = D.keyed(pl.read_parquet(os.path.join(DIAG, "map_control_de_mirage_stride10.parquet")))
    mc = mc.unique(KEY, keep="first", maintain_order=True)
    dx = D.keyed(pl.read_parquet(os.path.join(DIAG, "defuse_race_de_mirage.parquet")).select(KEY + ["nearest_ct_dist"]))
    dx = dx.unique(KEY, keep="first", maintain_order=True)
    G = (G.join(mc, on=KEY, how="left").join(dx, on=KEY, how="left")
          .join(D.plant_sites().select(["match_key", "round", "site"]), on=["match_key", "round"], how="left"))
    # Same mask as defuse_race_eval.py: a margin needs a plant position (site), so DX here is the pinned DX arm.
    planted = (pl.col("bomb_planted") == 1) & pl.col("seconds_since_plant").is_not_null() & pl.col("site").is_not_null()
    fill = pl.col("nearest_ct_dist").fill_null(MISSING).fill_nan(MISSING).clip(upper_bound=MISSING)
    G = G.with_columns(
        pl.when(planted).then(D.FUSE - pl.col("seconds_since_plant") - (fill / D.SPEED + D.DEFUSE)).otherwise(None).alias("m_DX")
    ).sort(KEY)
    n_na = G.filter(pl.col("mc_ct").is_null()).height
    print(f"rows {G.height:,}; snapshots without map control {n_na:,} ({time.time() - t0:.0f} s)", flush=True)
    arms = {"A": feats, "MC": feats + MC, "DX": feats + ["m_DX"], "DXMC": feats + ["m_DX"] + MC}

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
        per_seed[seed] = {"MC_minus_A": paired(match, y, preds["MC"], preds["A"]),
                          "DX_minus_A": paired(match, y, preds["DX"], preds["A"]),
                          "DXMC_minus_A": paired(match, y, preds["DXMC"], preds["A"]),
                          "DXMC_minus_DX": paired(match, y, preds["DXMC"], preds["DX"])}
        print(f"seed {seed}: MC-A {per_seed[seed]['MC_minus_A']['diff']:+.6f}  DX-A {per_seed[seed]['DX_minus_A']['diff']:+.6f}  "
              f"DXMC-DX {per_seed[seed]['DXMC_minus_DX']['diff']:+.6f} ({time.time() - t0:.0f} s)", flush=True)

    def summarize(name):
        d = [per_seed[s][name]["diff"] for s in SEEDS]
        ci = [per_seed[s][name]["ci95"] for s in SEEDS]
        mean = float(np.mean(d))
        return {"mean_over_seeds": mean, "per_seed": d, "per_seed_ci95": ci,
                "passes": bool(mean <= -0.0005 and all(c[1] < 0 for c in ci))}

    verdicts = {"map_control_adds_information": summarize("MC_minus_A"),
                "map_control_adds_beyond_DX": summarize("DXMC_minus_DX")}
    res = {"rows_test": int(te_ref.height), "rows_without_map_control": int(n_na), "seeds": SEEDS, "per_seed": per_seed,
           "verdicts": verdicts, "all_diffs_post_hoc": {k: summarize(k) for k in per_seed[SEEDS[0]].keys()},
           "thresholds": "mean(diff) <= -0.0005 and every seed's CI upper bound < 0", "seconds": round(time.time() - t0)}
    os.makedirs(DIAG, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: {"mean": v["mean_over_seeds"], "passes": v["passes"]} for k, v in verdicts.items()}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
