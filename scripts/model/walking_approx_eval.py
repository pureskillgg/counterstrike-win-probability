"""Does the centroid approximation change the walking results? Centroid versus portal-midpoint walking distances on the same rows
(de_mirage 2026, frozen split, every 10th second, three seeds, paired match-cluster bootstrap against the 57-input baseline).

Arms (57 inputs plus the listed features):
  A      57 baseline inputs
  S      + straight-line distance to the two sites, CT and T (4)
  Pc     + walking distance to the two sites, centroid-to-centroid path (4)          (nav_path_de_mirage_stride10.parquet)
  Pp     + walking distance to the two sites, portal-midpoint path (4)               (nav_portal_de_mirage_stride10.parquet)
  DX     + straight-line margin to the EXACT plant (10 s defuse, 250 units/s)
  DWXc   + walking margin to the EXACT plant, centroid path                          (nav_plant_de_mirage_stride10.parquet)
  DWXp   + walking margin to the EXACT plant, portal path                            (nav_portal_de_mirage_stride10.parquet)
  DWc    + walking margin to the planted SITE centre, centroid path
  DWp    + walking margin to the planted SITE centre, portal path

Pre-registered thresholds (written before any result was read):
  P_portal adds information     mean(Pp - A) <= -0.0005 and every seed's 95% CI upper bound < 0
  portal beats straight         mean(Pp - S) <= -0.0005 and every seed's 95% CI upper bound < 0
  DWXp adds information         mean(DWXp - A) <= -0.0005 and every seed's 95% CI upper bound < 0
  portal beats straight, exact  mean(DWXp - DX) <= -0.0005 and every seed's 95% CI upper bound < 0
  approximation matters         any of: |mean(Pp - Pc)| >= 0.0005, |mean(DWXp - DWXc)| >= 0.0005, |mean(DWp - DWc)| >= 0.0005, or a
                                pass/fail verdict in the list above differs between the centroid and portal versions
Match-cluster bootstrap: spatial_feature_eval.paired (1,000 resamples, seed 0). MISSING = 1e4 for unreachable or no-living-player seconds.

Writes data/diagnostics/walking_approx_eval.json.
Run in the cs2gpu environment (polars, xgboost):
  C:/Users/<user>/.pixi/workspace-envs/cs2gpu-15282324412032665829/envs/default/python.exe scripts/model/walking_approx_eval.py
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
OUT = os.path.join(DIAG, "walking_approx_eval.json")
KEY = ["match_key", "round", "second"]
SEEDS = [0, 1, 2]
MISSING = D.MISSING
CENTROID = ["ct_path_siteA", "ct_path_siteB", "t_path_siteA", "t_path_siteB"]
PORTAL = [c + "_p" for c in CENTROID]
STRAIGHT = D.STRAIGHT


def fill(col):
    return pl.col(col).fill_null(MISSING).fill_nan(MISSING).clip(upper_bound=MISSING)


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
    nav = D.keyed(pl.read_parquet(os.path.join(DIAG, "nav_path_de_mirage_stride10.parquet")))
    por = D.keyed(pl.read_parquet(os.path.join(DIAG, "nav_portal_de_mirage_stride10.parquet")))
    plant = D.keyed(pl.read_parquet(os.path.join(DIAG, "nav_plant_de_mirage_stride10.parquet"))).unique(KEY, keep="first", maintain_order=True)
    dx = D.keyed(pl.read_parquet(os.path.join(DIAG, "defuse_race_de_mirage.parquet")).select(KEY + ["nearest_ct_dist"]))
    dx = dx.unique(KEY, keep="first", maintain_order=True)
    sites = D.plant_sites()
    G = (G.join(nav, on=KEY, how="left").join(por, on=KEY, how="left").join(plant, on=KEY, how="left")
          .join(dx, on=KEY, how="left").join(sites.select(["match_key", "round", "site"]), on=["match_key", "round"], how="left"))

    G = G.with_columns([fill(c).alias(f"{c}_f")
                        for c in CENTROID + PORTAL + STRAIGHT + ["nearest_ct_dist", "ct_path_plant", "ct_path_plant_p"]])

    planted = (pl.col("bomb_planted") == 1) & pl.col("seconds_since_plant").is_not_null() & pl.col("site").is_not_null()
    fuse = D.FUSE - pl.col("seconds_since_plant")
    none = pl.lit(None, dtype=pl.Float64)

    def margin(travel):
        return pl.when(planted).then(fuse - (travel / D.SPEED + D.DEFUSE)).otherwise(none)

    def site_sel(prefix_a, prefix_b):
        return pl.when(pl.col("site") == 0).then(pl.col(prefix_a)).otherwise(pl.col(prefix_b))

    G = G.with_columns(
        margin(pl.col("nearest_ct_dist_f")).alias("m_DX"),
        margin(pl.col("ct_path_plant_f")).alias("m_DWXc"),
        margin(pl.col("ct_path_plant_p_f")).alias("m_DWXp"),
        margin(site_sel("ct_path_siteA_f", "ct_path_siteB_f")).alias("m_DWc"),
        margin(site_sel("ct_path_siteA_p_f", "ct_path_siteB_p_f")).alias("m_DWp"),
    ).sort(KEY)

    arms = {
        "A": feats,
        "S": feats + [c + "_f" for c in STRAIGHT],
        "Pc": feats + [c + "_f" for c in CENTROID],
        "Pp": feats + [c + "_f" for c in PORTAL],
        "DX": feats + ["m_DX"],
        "DWXc": feats + ["m_DWXc"],
        "DWXp": feats + ["m_DWXp"],
        "DWc": feats + ["m_DWc"],
        "DWp": feats + ["m_DWp"],
    }
    n_planted_test = G.filter((pl.col("split") == "test") & planted).height
    print(f"rows (stride 10): {G.height:,}; planted test rows {n_planted_test:,} ({time.time() - t0:.0f} s)", flush=True)

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
        ps = {f"{n}_minus_A": paired(match, y, preds[n], preds["A"]) for n in arms if n != "A"}
        ps["Pp_minus_Pc"] = paired(match, y, preds["Pp"], preds["Pc"])
        ps["Pp_minus_S"] = paired(match, y, preds["Pp"], preds["S"])
        ps["DWXp_minus_DX"] = paired(match, y, preds["DWXp"], preds["DX"])
        ps["DWXp_minus_DWXc"] = paired(match, y, preds["DWXp"], preds["DWXc"])
        ps["DWp_minus_DWc"] = paired(match, y, preds["DWp"], preds["DWc"])
        per_seed[seed] = ps
        print(f"seed {seed}: Pp-A {ps['Pp_minus_A']['diff']:+.6f}  Pc-A {ps['Pc_minus_A']['diff']:+.6f}  "
              f"DWXp-DWXc {ps['DWXp_minus_DWXc']['diff']:+.6f}  DWp-DWc {ps['DWp_minus_DWc']['diff']:+.6f} "
              f"({time.time() - t0:.0f} s)", flush=True)

    def summarize(name):
        d = [per_seed[s][name]["diff"] for s in SEEDS]
        ci = [per_seed[s][name]["ci95"] for s in SEEDS]
        mean = float(np.mean(d))
        return {"mean_over_seeds": mean, "per_seed": d, "per_seed_ci95": ci,
                "passes": bool(mean <= -0.0005 and all(c[1] < 0 for c in ci))}

    table = {k: summarize(k) for k in per_seed[SEEDS[0]].keys()}
    verdicts = {"P_portal_adds_information": table["Pp_minus_A"], "portal_beats_straight": table["Pp_minus_S"],
                "DWXp_adds_information": table["DWXp_minus_A"], "portal_beats_straight_exact": table["DWXp_minus_DX"]}
    mismatch = {
        "P_pass_differs": table["Pp_minus_A"]["passes"] != table["Pc_minus_A"]["passes"],
        "DWX_pass_differs": table["DWXp_minus_A"]["passes"] != table["DWXc_minus_A"]["passes"],
    }
    matters = bool(abs(table["Pp_minus_Pc"]["mean_over_seeds"]) >= 0.0005 or abs(table["DWXp_minus_DWXc"]["mean_over_seeds"]) >= 0.0005
                   or abs(table["DWp_minus_DWc"]["mean_over_seeds"]) >= 0.0005 or mismatch["P_pass_differs"] or mismatch["DWX_pass_differs"])
    res = {"planted_test_rows": n_planted_test, "seeds": SEEDS, "per_seed": per_seed, "verdicts": verdicts,
           "centroid_comparison_arms": {k: v for k, v in table.items() if k.startswith("Pc") or k.startswith("DWXc") or k.startswith("DWc")},
           "approximation_effect": {"Pp_minus_Pc": table["Pp_minus_Pc"], "DWXp_minus_DWXc": table["DWXp_minus_DWXc"],
                                    "DWp_minus_DWc": table["DWp_minus_DWc"]},
           "pass_fail_mismatch": mismatch, "approximation_matters": matters,
           "all_arms_post_hoc": table, "thresholds": "mean(diff) <= -0.0005 and every seed's CI upper bound < 0",
           "seconds": round(time.time() - t0)}
    os.makedirs(DIAG, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: {"mean": v["mean_over_seeds"], "passes": v["passes"]} for k, v in verdicts.items()}, indent=1), flush=True)
    print(f"approximation matters: {matters}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
