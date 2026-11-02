"""Extended defuse-race eval: the exact-plant margin over the navigation mesh, plus sensitivity and representation checks (de_mirage, frozen
split, every 10th second, three seeds, paired match-cluster bootstrap against the 57-input baseline).

Margin = fuse_left - (travel / SPEED + DEFUSE), fuse_left = 40 - seconds_since_plant, post-plant seconds only.
Travel to the EXACT plant position, for the nearest living CT:
  straight-line  nearest_ct_dist  (defuse_race_feature.py, DX)
  walking        ct_path_plant    (defuse_plant_nav.py: navigation-mesh shortest path to the plant's area, DWX)

Arms (all on the same rows; every arm is the 57 inputs plus one margin column unless noted):
  A       57 baseline inputs
  DX      straight-line, exact plant, 10 s defuse, 250 units/s          (must reproduce defuse_race_eval.py DX exactly)
  DS      straight-line, planted site centre                             (from defuse_race_eval.py)
  DW      walking, planted site centre                                    (from defuse_race_eval.py)
  DWX     walking, exact plant, 10 s defuse, 250 units/s                  (new: the paper's construction, navigation travel)
Sensitivity (post hoc, reported, not pass/fail):
  DX_d5, DWX_d5        defuse 5 s (defuser with a kit)
  DX_v200, DWX_v200    speed 200 units/s
  DX_v320, DWX_v320    speed 320 units/s
Representation (post hoc):
  DX_nan, DWX_nan      unreachable or no-CT seconds left as NaN (XGBoost default branch), not the MISSING value 1e4
  DX_mono, DWX_mono    margin column constrained to +1 (CT chance rises with margin)

Pre-registered thresholds for the new arms (written before any result was read):
  DWX adds information     mean(DWX - A) <= -0.0005 and every seed's 95% CI upper bound < 0
  walking beats straight   mean(DWX - DX) <= -0.0005 and every seed's 95% CI upper bound < 0
  DX reproduces            DX - A per seed equals defuse_race_eval.json to 1e-6 (determinism check)
Sensitivity and representation arms are reported against DX and DWX; no verdict is set for them.
Match-cluster bootstrap: spatial_feature_eval.paired (1,000 resamples, seed 0). MISSING = 1e4 unless the arm is a NaN arm.

Writes data/diagnostics/defuse_race_eval2.json.
Run in the cs2gpu environment (polars, xgboost):
  C:/Users/<user>/.pixi/workspace-envs/cs2gpu-15282324412032665829/envs/default/python.exe scripts/model/defuse_race_eval2.py
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
OUT = os.path.join(DIAG, "defuse_race_eval2.json")
KEY = ["match_key", "round", "second"]
SEEDS = [0, 1, 2]
MISSING = D.MISSING
PATH = D.PATH
STRAIGHT = D.STRAIGHT


def fill(col):
    return pl.col(col).fill_null(MISSING).fill_nan(MISSING).clip(upper_bound=MISSING)


def raw(col):
    return pl.when(pl.col(col).is_infinite()).then(None).otherwise(pl.col(col))


def fit_arm(G, cols, seed, positive):
    tr = G.filter(pl.col("split") == "train")
    va = G.filter(pl.col("split") == "val")
    te = G.filter(pl.col("split") == "test")
    mono = W.mono_for(cols)
    for c in positive:
        mono[cols.index(c)] = 1
    m, best = W.fit_xgb(tr.select(cols).to_numpy().astype(np.float32), tr["label_ct_win"].to_numpy().astype(np.int64),
                        va.select(cols).to_numpy().astype(np.float32), va["label_ct_win"].to_numpy().astype(np.int64),
                        mono=mono, seed=seed, device="cuda")
    return te, W.predict(m, best, te.select(cols).to_numpy().astype(np.float32))


def main():
    t0 = time.time()
    feats = json.load(open(os.path.join(ROOT, "models", "winprob_v6_drop_heavy.meta.json"), encoding="utf-8"))["features"]
    assert len(feats) == 57, len(feats)
    G = D.base_rows(feats)
    nav = D.keyed(pl.read_parquet(os.path.join(DIAG, "nav_path_de_mirage_stride10.parquet")))
    dx = D.keyed(pl.read_parquet(os.path.join(DIAG, "defuse_race_de_mirage.parquet")).select(KEY + ["nearest_ct_dist"]))
    dx = dx.unique(KEY, keep="first", maintain_order=True)
    plant = D.keyed(pl.read_parquet(os.path.join(DIAG, "nav_plant_de_mirage_stride10.parquet")))
    plant = plant.unique(KEY, keep="first", maintain_order=True)
    sites = D.plant_sites()
    G = (G.join(nav, on=KEY, how="left").join(dx, on=KEY, how="left").join(plant, on=KEY, how="left")
          .join(sites.select(["match_key", "round", "site"]), on=["match_key", "round"], how="left"))

    G = G.with_columns([fill(c).alias(f"{c}_f") for c in PATH + STRAIGHT + ["nearest_ct_dist", "ct_path_plant"]]
                       + [raw("nearest_ct_dist").alias("nearest_raw"), raw("ct_path_plant").alias("plant_raw")])

    planted = (pl.col("bomb_planted") == 1) & pl.col("seconds_since_plant").is_not_null() & pl.col("site").is_not_null()
    fuse = D.FUSE - pl.col("seconds_since_plant")
    none = pl.lit(None, dtype=pl.Float64)

    def margin(travel, defuse=D.DEFUSE, speed=D.SPEED):
        return pl.when(planted).then(fuse - (travel / speed + defuse)).otherwise(none)

    ct_path_sel = pl.when(pl.col("site") == 0).then(pl.col("ct_path_siteA_f")).otherwise(pl.col("ct_path_siteB_f"))
    st_ct_sel = pl.when(pl.col("site") == 0).then(pl.col("st_ct_siteA_f")).otherwise(pl.col("st_ct_siteB_f"))
    G = G.with_columns(
        margin(pl.col("nearest_ct_dist_f")).alias("m_DX"),
        margin(pl.col("ct_path_plant_f")).alias("m_DWX"),
        margin(st_ct_sel).alias("m_DS"),
        margin(ct_path_sel).alias("m_DW"),
        margin(pl.col("nearest_ct_dist_f"), defuse=5.0).alias("m_DX_d5"),
        margin(pl.col("ct_path_plant_f"), defuse=5.0).alias("m_DWX_d5"),
        margin(pl.col("nearest_ct_dist_f"), speed=200.0).alias("m_DX_v200"),
        margin(pl.col("ct_path_plant_f"), speed=200.0).alias("m_DWX_v200"),
        margin(pl.col("nearest_ct_dist_f"), speed=320.0).alias("m_DX_v320"),
        margin(pl.col("ct_path_plant_f"), speed=320.0).alias("m_DWX_v320"),
        margin(pl.col("nearest_raw")).alias("m_DX_nan"),
        margin(pl.col("plant_raw")).alias("m_DWX_nan"),
    ).sort(KEY)

    arms = {
        "A": (feats, []),
        "DX": (feats + ["m_DX"], []), "DS": (feats + ["m_DS"], []), "DW": (feats + ["m_DW"], []), "DWX": (feats + ["m_DWX"], []),
        "DX_d5": (feats + ["m_DX_d5"], []), "DWX_d5": (feats + ["m_DWX_d5"], []),
        "DX_v200": (feats + ["m_DX_v200"], []), "DWX_v200": (feats + ["m_DWX_v200"], []),
        "DX_v320": (feats + ["m_DX_v320"], []), "DWX_v320": (feats + ["m_DWX_v320"], []),
        "DX_nan": (feats + ["m_DX_nan"], []), "DWX_nan": (feats + ["m_DWX_nan"], []),
        "DX_mono": (feats + ["m_DX"], ["m_DX"]), "DWX_mono": (feats + ["m_DWX"], ["m_DWX"]),
    }
    n_planted_test = G.filter((pl.col("split") == "test") & planted).height
    print(f"rows (stride 10): {G.height:,}; planted test rows {n_planted_test:,}; "
          f"unreachable exact-plant seconds {G.filter(pl.col('ct_path_plant').is_infinite()).height:,} "
          f"({time.time() - t0:.0f} s)", flush=True)

    per_seed = {}
    te_ref = None
    for seed in SEEDS:
        preds = {}
        for name, (cols, positive) in arms.items():
            te_k, p = fit_arm(G, cols, seed, positive)
            preds[name] = p
            te_ref = te_k if te_ref is None else te_ref
        match = te_ref["match_key"].to_numpy()
        y = te_ref["label_ct_win"].to_numpy().astype(np.int64)
        ps = {}
        for name in arms:
            if name != "A":
                ps[f"{name}_minus_A"] = paired(match, y, preds[name], preds["A"])
        ps["DWX_minus_DX"] = paired(match, y, preds["DWX"], preds["DX"])
        ps["DWX_minus_DS"] = paired(match, y, preds["DWX"], preds["DS"])
        for name in ["DX_nan", "DWX_nan", "DX_mono", "DWX_mono"]:
            base = "DX" if name.startswith("DX") else "DWX"
            ps[f"{name}_minus_{base}"] = paired(match, y, preds[name], preds[base])
        per_seed[seed] = ps
        print(f"seed {seed}: DWX-A {ps['DWX_minus_A']['diff']:+.6f}  DX-A {ps['DX_minus_A']['diff']:+.6f}  "
              f"DWX-DX {ps['DWX_minus_DX']['diff']:+.6f}  ({time.time() - t0:.0f} s)", flush=True)

    def summarize(name):
        d = [per_seed[s][name]["diff"] for s in SEEDS]
        ci = [per_seed[s][name]["ci95"] for s in SEEDS]
        mean = float(np.mean(d))
        return {"mean_over_seeds": mean, "per_seed": d, "per_seed_ci95": ci,
                "passes": bool(mean <= -0.0005 and all(c[1] < 0 for c in ci))}

    names = [k for k in per_seed[SEEDS[0]].keys()]
    table = {k: summarize(k) for k in names}

    prev = json.load(open(os.path.join(DIAG, "defuse_race_eval.json"), encoding="utf-8"))["per_seed"]
    reproduce = [abs(per_seed[s]["DX_minus_A"]["diff"] - prev[str(s)]["DX_minus_A"]["diff"]) for s in SEEDS]
    verdicts = {"DWX_adds_information": table["DWX_minus_A"], "walking_beats_straight_exact": table["DWX_minus_DX"]}
    res = {"planted_test_rows": n_planted_test, "seeds": SEEDS, "per_seed": per_seed,
           "verdicts": verdicts, "all_arms_summary_post_hoc": table,
           "dx_reproduction_max_abs_diff": float(max(reproduce)),
           "dx_reproduces_defuse_race_eval": bool(max(reproduce) <= 1e-6),
           "thresholds": "mean(diff) <= -0.0005 and every seed's CI upper bound < 0",
           "seconds": round(time.time() - t0)}
    os.makedirs(DIAG, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: {"mean": v["mean_over_seeds"], "passes": v["passes"]} for k, v in verdicts.items()}, indent=1), flush=True)
    print(f"DX reproduces defuse_race_eval.json: {res['dx_reproduces_defuse_race_eval']} "
          f"(max abs diff {res['dx_reproduction_max_abs_diff']:.2e})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
