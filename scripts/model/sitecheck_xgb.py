"""Site-centre check, step 2: all-maps XGBoost with three arms and three seeds, saving predictions.

Arms (frozen recipe, W.fit_xgb, early stopping on the all-maps validation split; the 57 frozen inputs in every arm):
  A    57 inputs only
  SO   A + straight-line distances from the ORIGINAL centres (fitted on all matches, as in allmaps_xgb_geo.py)
  SN   A + straight-line distances from the TRAIN-ONLY centres (sitecheck_geo.py)
Seeds 0, 1, 2. Walking distances are not in this check (their arrays still use the original centres).

DECISION (pre-registered, written before any run; validation seconds only): the distance claim survives the centre check if
  (a) for EACH seed, SN minus A on the validation seconds has a mean <= -0.0005 and a 95% match-cluster CI upper bound < 0, and
  (b) the three-seed mean of SN minus A on the validation seconds is <= -0.0005.
The test seconds (the de_mirage test, 865,974 seconds) are scored and their predictions saved for the record. They play no part in
the decision. SO minus A on validation is reported beside it as the reference the original claim came from.
Caveat, applies to every arm alike: the validation split is also the early-stopping set, so it is not a clean decision set in the
strict sense. The bias is the same for A, SO and SN, so the paired difference is still informative about the centres.

Saves predictions (validation: all maps; test: de_mirage) to data/diagnostics/preds/allmaps_xgb_<arm>_seed<s>.parquet, and the
decision to data/diagnostics/allmaps_xgb_sitecheck.json.

Run in the cs2gpu environment (polars, xgboost, CUDA):
  C:/Users/13144/.pixi/workspace-envs/cs2gpu-15282324412032665829/envs/default/python.exe -I -X utf8 -u scripts/model/sitecheck_xgb.py
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
PREDS = os.path.join(DIAG, "preds")
OUT = os.path.join(DIAG, "allmaps_xgb_sitecheck.json")
MAPS = ["de_mirage", "de_dust2", "de_ancient", "de_anubis", "de_inferno", "de_nuke", "de_cache"]
KEY = ["match_key", "round", "second"]
ST = [f"st{i}" for i in range(4)]
NST = [f"nst{i}" for i in range(4)]
SEEDS = [0, 1, 2]


def main():
    t0 = time.time()
    feats = json.load(open(os.path.join(ROOT, "models", "winprob_v6_drop_heavy.meta.json"), encoding="utf-8"))["features"]
    assert len(feats) == 57, len(feats)
    geo_old = pl.concat([pl.read_parquet(os.path.join(DIAG, "allmaps_geo", f"{m}.parquet")) for m in MAPS])
    geo_new = pl.concat([pl.read_parquet(os.path.join(DIAG, "allmaps_geo_trainsites", f"{m}.parquet")) for m in MAPS])
    geo_new = geo_new.rename({c: "n" + c for c in ST})
    C = pl.read_parquet(os.path.join(ROOT, "data", "combined", "cache_snapshot_2026_ext.parquet"),
                        columns=sorted(set(KEY + ["match_key", "label_ct_win", "map_name"] + feats)))
    C = W.drop_roster_defect_rounds(C)
    C = C.filter(pl.col("label_ct_win").is_not_null())
    C = C.with_columns(pl.col("round").cast(pl.Int64), pl.col("second").cast(pl.Int64))
    man = W.load_manifest(os.path.join(ROOT, "data", "combined", "split_manifest.json"))["assign"]
    A = pl.DataFrame({"match_key": list(man.keys()), "split": list(man.values())})
    G = C.join(geo_old, on=KEY, how="inner").join(geo_new, on=KEY, how="inner").join(A, on="match_key", how="inner")
    G = G.unique(KEY, keep="first", maintain_order=True).sort(KEY)
    print(f"joined rows {G.height:,} ({time.time() - t0:.0f} s)", flush=True)
    tr = G.filter(pl.col("split") == "train")
    va = G.filter(pl.col("split") == "val")
    te = G.filter((pl.col("split") == "test") & (pl.col("map_name") == "de_mirage"))
    y_va = va["label_ct_win"].to_numpy().astype(np.int64)
    y_te = te["label_ct_win"].to_numpy().astype(np.int64)
    match_va = va["match_key"].to_numpy()
    match_te = te["match_key"].to_numpy()
    arms = {"A": feats, "SO": feats + ST, "SN": feats + NST}
    os.makedirs(PREDS, exist_ok=True)

    per_seed = []
    for seed in SEEDS:
        preds_va, preds_te = {}, {}
        for name, cols in arms.items():
            m, best = W.fit_xgb(tr.select(cols).to_numpy().astype(np.float32), tr["label_ct_win"].to_numpy().astype(np.int64),
                                va.select(cols).to_numpy().astype(np.float32), va["label_ct_win"].to_numpy().astype(np.int64),
                                mono=W.mono_for(cols), seed=seed, device="cuda")
            preds_va[name] = W.predict(m, best, va.select(cols).to_numpy().astype(np.float32))
            preds_te[name] = W.predict(m, best, te.select(cols).to_numpy().astype(np.float32))
            print(f"seed {seed} arm {name}: validation log loss {float(W.ll_vec(preds_va[name], y_va).mean()):.6f}, "
                  f"test log loss {float(W.ll_vec(preds_te[name], y_te).mean()):.6f}, best iteration {best} "
                  f"({time.time() - t0:.0f} s)", flush=True)
            frame_va = pl.DataFrame({"match_key": match_va, "round": va["round"].to_numpy(), "second": va["second"].to_numpy(),
                                     "map_name": va["map_name"].to_numpy(), "split": "val", "label": y_va,
                                     "p": preds_va[name].astype(np.float64)})
            frame_te = pl.DataFrame({"match_key": match_te, "round": te["round"].to_numpy(), "second": te["second"].to_numpy(),
                                     "map_name": te["map_name"].to_numpy(), "split": "test", "label": y_te,
                                     "p": preds_te[name].astype(np.float64)})
            pl.concat([frame_va, frame_te]).write_parquet(os.path.join(PREDS, f"allmaps_xgb_{name}_seed{seed}.parquet"))
        row = {"seed": seed,
               "validation_log_loss": {n: float(W.ll_vec(p, y_va).mean()) for n, p in preds_va.items()},
               "test_log_loss": {n: float(W.ll_vec(p, y_te).mean()) for n, p in preds_te.items()},
               "SN_minus_A_validation": paired(match_va, y_va, preds_va["SN"], preds_va["A"]),
               "SO_minus_A_validation": paired(match_va, y_va, preds_va["SO"], preds_va["A"]),
               "SN_minus_A_test_descriptive": paired(match_te, y_te, preds_te["SN"], preds_te["A"]),
               "SO_minus_A_test_descriptive": paired(match_te, y_te, preds_te["SO"], preds_te["A"])}
        d = row["SN_minus_A_validation"]
        row["SN_passes_validation"] = bool(d["diff"] <= -0.0005 and d["ci95"][1] < 0)
        per_seed.append(row)
        print(json.dumps({k: row[k] for k in ["seed", "SN_passes_validation"]}), flush=True)

    mean_sn = float(np.mean([r["SN_minus_A_validation"]["diff"] for r in per_seed]))
    mean_so = float(np.mean([r["SO_minus_A_validation"]["diff"] for r in per_seed]))
    decision = bool(all(r["SN_passes_validation"] for r in per_seed) and mean_sn <= -0.0005)
    res = {"seeds": per_seed, "mean_SN_minus_A_validation": mean_sn, "mean_SO_minus_A_validation": mean_so,
           "distance_claim_survives_centre_check": decision, "decision_set": "validation seconds only",
           "seconds": round(time.time() - t0)}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ["mean_SN_minus_A_validation", "mean_SO_minus_A_validation",
                                          "distance_claim_survives_centre_check"]}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
