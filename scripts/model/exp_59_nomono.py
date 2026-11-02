"""Monotone constraints for the winning model (59 inputs): the same 59 inputs with no monotone constraints, paired against the saved
winprob_v4 fit (which has them) on the same split and test seconds. Settings as winprob_v4; only the constraints differ.

  ...python.exe -I -X utf8 scripts\\model\\exp_59_nomono.py
Writes data/combined/exp_59_nomono.json.
"""
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import polars as pl
import xgboost as xgb

import wp_common as W

sys.stdout.reconfigure(encoding="utf-8")


def main():
    t0 = time.time()
    store = "combined"
    d = os.path.join(W.ROOT, "data", store)
    feats59 = json.load(open(os.path.join(W.ROOT, "models", "winprob_v3.meta.json"), encoding="utf-8"))["features"]
    ref_meta = json.load(open(os.path.join(W.ROOT, "models", "winprob_v4.meta.json"), encoding="utf-8"))
    assert list(ref_meta["features"]) == list(feats59), "feature order differs from the reference model"

    need = list(dict.fromkeys(["match_key", "round", "tick", "second", "label_ct_win", "w_round"] + feats59))
    lf = pl.scan_parquet(os.path.join(d, "cache_snapshot_2026.parquet")).select(need)
    kdf = lf.select("match_key").unique().collect().with_columns(pl.lit(1.0).alias("w_round"), pl.lit(0).alias("label_ct_win"),
                                                                 pl.lit("2026").alias("era"), pl.lit("").alias("match_date"))
    sp = W.make_split("random", kdf, seed=0)
    del kdf
    G = lf.with_columns(pl.col(feats59).cast(pl.Float32)).collect(engine="streaming")
    tr, va, te = sorted(sp["train"]), sorted(sp["val"]), sorted(sp["test"])
    _, Xtr, ytr = W.arrays(G, tr, feats59, stride=1)
    sub_va, Xva, yva = W.arrays(G, va, feats59, stride=1)
    sub_te, Xte, yte = W.arrays(G, te, feats59, stride=1)
    del G
    gc.collect()
    print(f"train {len(ytr):,}, validation {len(yva):,}, test {len(yte):,} ({time.time() - t0:.0f}s)", flush=True)

    ref = xgb.Booster()
    ref.load_model(os.path.join(W.ROOT, "models", "winprob_v4.json"))
    best_ref = int(ref_meta["best_iteration"])
    p_ref = ref.predict(xgb.DMatrix(Xte), iteration_range=(0, best_ref + 1))
    pva_ref = ref.predict(xgb.DMatrix(Xva), iteration_range=(0, best_ref + 1))

    m, best = W.fit_xgb(Xtr, ytr, Xva, yva, [0] * len(feats59), seed=0, device="cuda")
    p = W.predict(m, best, Xte)
    pva = W.predict(m, best, Xva)
    print(f"59 inputs, no monotone constraints: best iteration {best} ({time.time() - t0:.0f}s)", flush=True)

    ev = W.evaluate(sub_te, yte, p, ref=p_ref)
    evv = W.evaluate(sub_va, yva, pva, ref=pva_ref)
    out = {
        "reference": "winprob_v4 (59 inputs, monotone constraints on the alive counts), same rows",
        "reference_best_iteration": best_ref, "no_mono_best_iteration": int(best),
        "test_logloss_no_mono": ev["logloss"], "test_logloss_no_mono_ci95": ev["logloss_ci95"],
        "test_delta_no_mono_minus_mono": ev["delta_vs_ref"], "test_delta_ci95": ev["delta_vs_ref_ci95"],
        "validation_logloss_no_mono": evv["logloss"],
        "validation_delta_no_mono_minus_mono": evv["delta_vs_ref"], "validation_delta_ci95": evv["delta_vs_ref_ci95"],
        "test_matches": ev["n_matches"], "validation_matches": evv["n_matches"],
    }
    W.dump_json(out, os.path.join(d, "exp_59_nomono.json"))
    print(f"no-mono minus mono: test {ev['delta_vs_ref']:+.5f} [{ev['delta_vs_ref_ci95'][0]:+.5f}, {ev['delta_vs_ref_ci95'][1]:+.5f}]; "
          f"validation {evv['delta_vs_ref']:+.5f} [{evv['delta_vs_ref_ci95'][0]:+.5f}, {evv['delta_vs_ref_ci95'][1]:+.5f}] "
          f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
