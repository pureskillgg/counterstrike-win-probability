"""Validation check for the pooled model with round-type inputs (V1), against one model for every round (M1).

  ...python.exe -I -X utf8 scripts\\model\\exp_pistol_val.py [--store combined] [--device cuda]

Both models use the same settings and rows as exp_pistol.py. The comparison is on the validation matches, which the rule in the
round-method run uses to decide, so the test numbers in exp_pistol.json are not used for the decision here.
Writes data/<store>/exp_pistol_validation.json.
"""
import argparse
import json
import os
import sys
import time

os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import polars as pl
import xgboost as xgb

import wp_common as W
from exp_pistol import AFTER, PISTOL, SMALL_GROUP, arrays, extras, train

sys.stdout.reconfigure(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="combined")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    a = ap.parse_args()
    d = os.path.join(W.ROOT, "data", a.store)
    t0 = time.time()
    feats = json.load(open(os.path.join(W.ROOT, "models", "winprob_v3.meta.json"), encoding="utf-8"))["features"]
    need = list(dict.fromkeys(["match_key", "round", "second", "label_ct_win", "w_round"] + feats))
    lf = pl.scan_parquet(os.path.join(d, "cache_snapshot_2026.parquet")).select(need)
    kdf = lf.select("match_key").unique().collect().with_columns(pl.lit(1.0).alias("w_round"), pl.lit(0).alias("label_ct_win"),
                                                                 pl.lit("2026").alias("era"), pl.lit("").alias("match_date"))
    sp = W.make_split("random", kdf, seed=0)
    tr, va = sorted(sp["train"]), sorted(sp["val"])
    mono = list(W.mono_for(feats))
    Xva, yva, rva, mkva, wva = arrays(lf, va, feats, False)
    va_sub = pl.DataFrame({"match_key": mkva.to_list(), "w_round": wva.to_list()})
    Xva_e = np.hstack([Xva, extras(pl.DataFrame({"round": rva}))])
    out = {}

    bst, best, s = train(lf, tr, va, feats, False, mono, None, False, {"seed": 0}, a.device)
    p_m1 = bst.predict(xgb.DMatrix(Xva), iteration_range=(0, best + 1))
    out["M1"] = {"best_iteration": best, "logloss": W.evaluate(va_sub, yva, p_m1)["logloss"]}
    print(f"M1 validation log loss {out['M1']['logloss']:.5f} (best {best}, {time.time() - t0:.0f}s)", flush=True)

    bst, best, s = train(lf, tr, va, feats, True, mono + [0, 0, 0, 0], None, False, {"seed": 0}, a.device)
    p_v1 = bst.predict(xgb.DMatrix(Xva_e), iteration_range=(0, best + 1))
    rv = W.evaluate(va_sub, yva, p_v1, ref=p_m1)
    out["V1"] = {"best_iteration": best, "logloss": rv["logloss"], "delta_vs_M1": rv["delta_vs_ref"], "delta_vs_M1_ci95": rv["delta_vs_ref_ci95"]}
    print(f"V1 validation log loss {rv['logloss']:.5f}, delta vs M1 {rv['delta_vs_ref']:+.5f} "
          f"[{rv['delta_vs_ref_ci95'][0]:+.5f}, {rv['delta_vs_ref_ci95'][1]:+.5f}] (best {best}, {time.time() - t0:.0f}s)", flush=True)

    groups = {"pistol": np.isin(rva, [1, 13]), "after": np.isin(rva, [2, 14]), "rest": ~np.isin(rva, [1, 13, 2, 14])}
    for g, m in groups.items():
        rg = W.evaluate(va_sub.filter(pl.Series(m)), yva[m], p_v1[m], ref=p_m1[m])
        out[f"V1_{g}"] = {"seconds": int(m.sum()), "delta_vs_M1": rg["delta_vs_ref"], "delta_vs_M1_ci95": rg["delta_vs_ref_ci95"]}
        print(f"  {g}: {rg['delta_vs_ref']:+.5f} [{rg['delta_vs_ref_ci95'][0]:+.5f}, {rg['delta_vs_ref_ci95'][1]:+.5f}]", flush=True)
    # V2 and V2b: the separate models, trained as in exp_pistol.py and scored on the validation seconds of their own group
    for name, group, mask in [("V2", PISTOL, np.isin(rva, [1, 13])), ("V2b", AFTER, np.isin(rva, [2, 14]))]:
        bst, best, s = train(lf, tr, va, feats, False, mono, group, False, dict(seed=0, **SMALL_GROUP), a.device)
        p = p_m1.copy()
        p[mask] = bst.predict(xgb.DMatrix(Xva[mask]), iteration_range=(0, best + 1))
        rv = W.evaluate(va_sub, yva, p, ref=p_m1)
        out[name] = {"best_iteration": best, "logloss": rv["logloss"], "delta_vs_M1": rv["delta_vs_ref"], "delta_vs_M1_ci95": rv["delta_vs_ref_ci95"]}
        rg = W.evaluate(va_sub.filter(pl.Series(mask)), yva[mask], p[mask], ref=p_m1[mask])
        out[f"{name}_group"] = {"seconds": int(mask.sum()), "delta_vs_M1": rg["delta_vs_ref"], "delta_vs_M1_ci95": rg["delta_vs_ref_ci95"]}
        print(f"{name} validation: group {rg['delta_vs_ref']:+.5f} [{rg['delta_vs_ref_ci95'][0]:+.5f}, {rg['delta_vs_ref_ci95'][1]:+.5f}]; "
              f"overall {rv['delta_vs_ref']:+.5f} (best {best}, {time.time() - t0:.0f}s)", flush=True)
    W.dump_json(out, os.path.join(d, "exp_pistol_validation.json"))
    print(f"done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
