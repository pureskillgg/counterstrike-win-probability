"""Defuse-race margin on the 57-input XGBoost (paired, de_mirage, frozen split, three seeds): the construction the paper uses, tested.

Margin = fuse_left - (travel + DEFUSE), with fuse_left = 40 - seconds_since_plant, DEFUSE = 10 s (no kit), travel = distance / 250 units
per second for the nearest LIVING CT. Post-plant seconds only; pre-plant seconds are missing, so XGBoost sends them down its default
branch. defuse_race_feature.py defines the margin with straight-line distance to the exact plant position.

Arms, all on the same rows (every 10th second of the frozen split, de_mirage 2026):
  A    57 baseline inputs (the baseline already has bomb_planted and seconds_since_plant)
  DX   A + margin, straight-line distance to the EXACT plant position (defuse_race_de_mirage.parquet)
  DS   A + margin, straight-line distance to the planted SITE centre (site approximation)
  DW   A + margin, walking distance over the navigation mesh to the planted SITE centre (same approximation as DS)
The site approximation assigns each round to the nearer of the two site centres (de_mirage_site_areas.json) using the plant position.
DS and DW differ only in the distance metric, so DW - DS is the walking-versus-straight comparison under the same approximation.

Pre-registered thresholds (written before any result was read):
  DW adds information     mean(DW - A) <= -0.0005 and every seed's 95% CI upper bound < 0
  DS adds information     mean(DS - A) <= -0.0005 and every seed's 95% CI upper bound < 0
  DX adds information     mean(DX - A) <= -0.0005 and every seed's 95% CI upper bound < 0
  walking beats straight  mean(DW - DS) <= -0.0005 and every seed's 95% CI upper bound < 0
  exact beats site        mean(DX - DS) <= -0.0005 and every seed's 95% CI upper bound < 0 (diagnostic of the site approximation)
Match-cluster bootstrap: spatial_feature_eval.paired (1,000 resamples). Unreachable or missing distances use MISSING = 1e4 (far), as in
nav_path_eval2.py; with fuse 40 s that makes the margin strongly negative (infeasible).

Writes data/diagnostics/defuse_race_eval.json.
Run in the cs2gpu environment (polars, xgboost):
  C:/Users/13144/.pixi/workspace-envs/cs2gpu-15282324412032665829/envs/default/python.exe scripts/model/defuse_race_eval.py
"""
import glob
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
OUT = os.path.join(DIAG, "defuse_race_eval.json")
KEY = ["match_key", "round", "second"]
STRIDE = 10
SEEDS = [0, 1, 2]
FUSE = 40.0
DEFUSE = 10.0
SPEED = 250.0
MISSING = 1.0e4
PATH = ["ct_path_siteA", "ct_path_siteB", "t_path_siteA", "t_path_siteB"]
STRAIGHT = ["st_ct_siteA", "st_ct_siteB", "st_t_siteA", "st_t_siteB"]


def keyed(df):
    return df.with_columns([pl.col("round").cast(pl.Int64), pl.col("second").cast(pl.Int64)])


def base_rows(feats):
    need = KEY + ["label_ct_win", "map_name", "bomb_planted", "seconds_since_plant"] + feats
    C = pl.read_parquet(os.path.join(ROOT, "data", "combined", "cache_snapshot_2026.parquet"), columns=sorted(set(need)))
    C = W.drop_roster_defect_rounds(C)
    C = C.filter((pl.col("map_name") == "de_mirage") & pl.col("label_ct_win").is_not_null() & (pl.col("second") % STRIDE == 0))
    man = W.load_manifest(os.path.join(ROOT, "data", "combined", "split_manifest.json"))["assign"]
    A = pl.DataFrame({"match_key": list(man.keys()), "split": list(man.values())})
    return keyed(C).join(A, on="match_key", how="inner").unique(KEY, keep="first", maintain_order=True).sort(KEY)


def plant_sites():
    files = sorted(glob.glob(os.path.join(ROOT, "data", "positions_1hz", "_plants", "2026-*.parquet")))
    P = pl.concat([
        pl.read_parquet(f).filter(pl.col("x").is_not_null() & pl.col("y").is_not_null())
          .group_by(["match_key", "round"]).agg(pl.col("x").first().alias("bx"), pl.col("y").first().alias("by"))
        for f in files])
    P = P.select(["match_key", "round", "bx", "by"]).with_columns(pl.col("round").cast(pl.Int64))
    P = P.group_by(["match_key", "round"]).agg(pl.col("bx").first(), pl.col("by").first())
    (ax, ay), (cx, cy) = json.load(open(os.path.join(ROOT, "data", "geometry", "de_mirage_site_areas.json"), encoding="utf-8"))["site_centres_xy"]
    d_a = (pl.col("bx") - ax) ** 2 + (pl.col("by") - ay) ** 2
    d_b = (pl.col("bx") - cx) ** 2 + (pl.col("by") - cy) ** 2
    return P.with_columns(pl.when(d_a <= d_b).then(0).otherwise(1).alias("site")).select(["match_key", "round", "bx", "by", "site"])


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
    nav = keyed(pl.read_parquet(os.path.join(DIAG, "nav_path_de_mirage_stride10.parquet")))
    dx = keyed(pl.read_parquet(os.path.join(DIAG, "defuse_race_de_mirage.parquet")).select(KEY + ["nearest_ct_dist"]))
    dx = dx.unique(KEY, keep="first", maintain_order=True)
    sites = plant_sites()
    G = G.join(nav, on=KEY, how="left").join(dx, on=KEY, how="left").join(sites.select(["match_key", "round", "site"]),
                                                                          on=["match_key", "round"], how="left")
    G = G.with_columns([pl.col(c).fill_null(MISSING).fill_nan(MISSING).clip(upper_bound=MISSING)
                        for c in PATH + STRAIGHT + ["nearest_ct_dist"]])
    planted = (pl.col("bomb_planted") == 1) & pl.col("seconds_since_plant").is_not_null() & pl.col("site").is_not_null()
    fuse_left = FUSE - pl.col("seconds_since_plant")
    ct_path_sel = pl.when(pl.col("site") == 0).then(pl.col("ct_path_siteA")).otherwise(pl.col("ct_path_siteB"))
    st_ct_sel = pl.when(pl.col("site") == 0).then(pl.col("st_ct_siteA")).otherwise(pl.col("st_ct_siteB"))
    none = pl.lit(None, dtype=pl.Float64)
    G = G.with_columns(
        pl.when(planted).then(fuse_left - (ct_path_sel / SPEED + DEFUSE)).otherwise(none).alias("m_DW"),
        pl.when(planted).then(fuse_left - (st_ct_sel / SPEED + DEFUSE)).otherwise(none).alias("m_DS"),
        pl.when(planted).then(fuse_left - (pl.col("nearest_ct_dist") / SPEED + DEFUSE)).otherwise(none).alias("m_DX"),
    ).sort(KEY)
    n_planted_test = G.filter((pl.col("split") == "test") & planted).height
    print(f"rows (stride {STRIDE}): {G.height:,}; planted test rows {n_planted_test:,}; "
          f"rows with site {G.filter(pl.col('site').is_not_null()).height:,} ({time.time() - t0:.0f} s)", flush=True)

    arms = {"A": feats, "DX": feats + ["m_DX"], "DS": feats + ["m_DS"], "DW": feats + ["m_DW"]}
    per_seed = {}
    te_ref = None
    for seed in SEEDS:
        preds = {}
        for name, cols in arms.items():
            te_k, p, _ = fit(G, cols, seed)
            preds[name] = p
            te_ref = te_k if te_ref is None else te_ref
        match = te_ref["match_key"].to_numpy()
        y = te_ref["label_ct_win"].to_numpy().astype(np.int64)
        per_seed[seed] = {"DX_minus_A": paired(match, y, preds["DX"], preds["A"]),
                          "DS_minus_A": paired(match, y, preds["DS"], preds["A"]),
                          "DW_minus_A": paired(match, y, preds["DW"], preds["A"]),
                          "DW_minus_DS": paired(match, y, preds["DW"], preds["DS"]),
                          "DX_minus_DS": paired(match, y, preds["DX"], preds["DS"])}
        print(f"seed {seed}: DW-A {per_seed[seed]['DW_minus_A']['diff']:+.6f}  DS-A {per_seed[seed]['DS_minus_A']['diff']:+.6f}  "
              f"DX-A {per_seed[seed]['DX_minus_A']['diff']:+.6f}  DW-DS {per_seed[seed]['DW_minus_DS']['diff']:+.6f}  "
              f"({time.time() - t0:.0f} s)", flush=True)

    def rule(name):
        d = [per_seed[s][name]["diff"] for s in SEEDS]
        ci = [per_seed[s][name]["ci95"] for s in SEEDS]
        mean = float(np.mean(d))
        return {"mean_over_seeds": mean, "per_seed": d, "per_seed_ci95": ci,
                "passes": bool(mean <= -0.0005 and all(c[1] < 0 for c in ci))}

    # Descriptive, post hoc: how often the margin sign agrees between arms, and CT win rate by margin sign on planted test rows.
    T = G.filter((pl.col("split") == "test") & planted)
    sign_agree = float(T.select(((pl.col("m_DW") > 0) == (pl.col("m_DS") > 0)).mean()).item())
    win_by_sign = {}
    for col in ["m_DX", "m_DS", "m_DW"]:
        pos = T.filter(pl.col(col) > 0)
        neg = T.filter(pl.col(col) <= 0)
        win_by_sign[col] = {"ct_win_rate_positive": float(pos["label_ct_win"].mean()) if pos.height else None,
                            "rows_positive": pos.height,
                            "ct_win_rate_nonpositive": float(neg["label_ct_win"].mean()) if neg.height else None,
                            "rows_nonpositive": neg.height}

    res = {"rows_test": int(te_ref.height), "test_matches": int(np.unique(te_ref["match_key"].to_numpy()).size),
           "planted_test_rows": n_planted_test, "seeds": SEEDS, "per_seed": per_seed,
           "verdicts": {"DW_adds_information": rule("DW_minus_A"), "DS_adds_information": rule("DS_minus_A"),
                        "DX_adds_information": rule("DX_minus_A"), "walking_beats_straight": rule("DW_minus_DS"),
                        "exact_beats_site": rule("DX_minus_DS")},
           "descriptive_post_hoc": {"margin_sign_agreement_DW_vs_DS_planted_test": sign_agree,
                                    "ct_win_rate_by_margin_sign_planted_test": win_by_sign},
           "thresholds": "mean(diff) <= -0.0005 and every seed's CI upper bound < 0",
           "seconds": round(time.time() - t0)}
    os.makedirs(DIAG, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: {"mean": v["mean_over_seeds"], "passes": v["passes"]} for k, v in res["verdicts"].items()}, indent=1), flush=True)
    print(json.dumps(res["descriptive_post_hoc"], indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
