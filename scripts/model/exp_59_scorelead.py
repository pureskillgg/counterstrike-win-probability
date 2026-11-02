"""Score lead on top of the 59 inputs (the winning model), paired on the same split and the same test seconds.

The reference is the saved winprob_v4 fit (59 inputs, GPU, seed 0, its own monotone constraints), reloaded and scored on the test rows.
The new model is the same 59 inputs plus score_lead_diag (freeze-end CT score minus T score, from the grid; unconstrained), with the same
settings and the same early-stopping rule on validation. Only the extra column differs.

  ...python.exe -I -X utf8 scripts\\model\\exp_59_scorelead.py
Writes data/combined/exp_59_scorelead.json.
"""
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
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
    mono59 = [int(ref_meta["monotone_constraints"][f]) for f in feats59]

    need = list(dict.fromkeys(["match_key", "round", "tick", "second", "label_ct_win", "w_round"] + feats59))
    lf = pl.scan_parquet(os.path.join(d, "cache_snapshot_2026.parquet")).select(need)
    kdf = lf.select("match_key").unique().collect().with_columns(pl.lit(1.0).alias("w_round"), pl.lit(0).alias("label_ct_win"),
                                                                 pl.lit("2026").alias("era"), pl.lit("").alias("match_date"))
    sp = W.make_split("random", kdf, seed=0)
    del kdf
    G = lf.with_columns(pl.col(feats59).cast(pl.Float32)).collect(engine="streaming")
    lead = (pl.scan_parquet(os.path.join(d, "grid.parquet")).filter(pl.col("era") == "2026")
            .select(["match_key", "round", "score_lead_diag"]).unique(subset=["match_key", "round"]).collect())
    G = G.join(lead, on=["match_key", "round"], how="left")
    missing_lead = int(G["score_lead_diag"].is_null().sum())
    G = G.with_columns(pl.col("score_lead_diag").fill_null(0).cast(pl.Float32))
    print(f"rows {G.height:,}; rows with no grid score lead: {missing_lead} ({time.time() - t0:.0f}s)", flush=True)

    tr, va, te = sorted(sp["train"]), sorted(sp["val"]), sorted(sp["test"])
    feats_s = feats59 + ["score_lead_diag"]
    _, Xtr, ytr = W.arrays(G, tr, feats_s, stride=1)
    sub_va, Xva, yva = W.arrays(G, va, feats_s, stride=1)
    sub_te, Xte, yte = W.arrays(G, te, feats_s, stride=1)
    del G
    gc.collect()
    print(f"train {len(ytr):,}, validation {len(yva):,}, test {len(yte):,} ({time.time() - t0:.0f}s)", flush=True)

    # reference: the saved winprob_v4 fit on the same rows (its inputs are the first 59 columns)
    ref = xgb.Booster()
    ref.load_model(os.path.join(W.ROOT, "models", "winprob_v4.json"))
    best_ref = int(ref_meta["best_iteration"])
    p_ref = ref.predict(xgb.DMatrix(Xte[:, :59]), iteration_range=(0, best_ref + 1))
    pva_ref = ref.predict(xgb.DMatrix(Xva[:, :59]), iteration_range=(0, best_ref + 1))
    ref_check = W.evaluate(sub_te, yte, p_ref)["logloss"]
    print(f"reference 59 inputs on these rows: test {ref_check:.6f} (meta {ref_meta['test_logloss']:.6f})", flush=True)

    m, best = W.fit_xgb(Xtr, ytr, Xva, yva, mono59 + [0], seed=0, device="cuda")
    p = W.predict(m, best, Xte)
    pva = W.predict(m, best, Xva)
    print(f"59 + score lead fitted: best iteration {best} ({time.time() - t0:.0f}s)", flush=True)

    ev = W.evaluate(sub_te, yte, p, ref=p_ref)
    evv = W.evaluate(sub_va, yva, pva, ref=pva_ref)
    out = {
        "reference": "winprob_v4 (59 inputs), reloaded and scored on the same rows",
        "reference_best_iteration": best_ref, "new_best_iteration": int(best),
        "monotone_59": mono59, "monotone_score_lead": 0,
        "rows_without_grid_score_lead": missing_lead,
        "train_rows": int(len(ytr)), "validation_rows": int(len(yva)), "test_rows": int(len(yte)),
        "test_matches": ev["n_matches"], "validation_matches": evv["n_matches"],
        "reference_test_logloss_recomputed": ref_check, "reference_test_logloss_meta": ref_meta["test_logloss"],
        "test_logloss_59_plus_score_lead": ev["logloss"], "test_logloss_59_plus_score_lead_ci95": ev["logloss_ci95"],
        "test_delta_vs_59": ev["delta_vs_ref"], "test_delta_vs_59_ci95": ev["delta_vs_ref_ci95"],
        "validation_logloss_59_plus_score_lead": evv["logloss"], "validation_delta_vs_59": evv["delta_vs_ref"],
        "validation_delta_vs_59_ci95": evv["delta_vs_ref_ci95"],
    }
    W.dump_json(out, os.path.join(d, "exp_59_scorelead.json"))
    print(f"test delta vs 59 inputs {ev['delta_vs_ref']:+.5f} [{ev['delta_vs_ref_ci95'][0]:+.5f}, {ev['delta_vs_ref_ci95'][1]:+.5f}]; "
          f"validation {evv['delta_vs_ref']:+.5f} [{evv['delta_vs_ref_ci95'][0]:+.5f}, {evv['delta_vs_ref_ci95'][1]:+.5f}] "
          f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
