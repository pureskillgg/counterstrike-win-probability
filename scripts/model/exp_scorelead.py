"""Score lead at the freeze end as a 14th input to the 13-input model. The column score_lead_diag is in grid.parquet (a freeze-end context
column, so it is known before the round starts). Same settings as winprob_v4_basic (depth 4, eta 0.05, min child weight 20, subsample 0.8,
early stopping after 100 rounds on validation, seed 0, monotone alive counts; score lead unconstrained), the same frozen split, scored on the
same test seconds and paired with the served model.

  ...python.exe -I -X utf8 scripts\\model\\exp_scorelead.py
Writes data/combined/exp_scorelead.json. CPU (xgboost hist), about 15 minutes.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import polars as pl
import xgboost as xgb

import train_final as T
import wp_common as W

sys.stdout.reconfigure(encoding="utf-8")


def main():
    t0 = time.time()
    store = "combined"
    cols = ["match_key", "round", "tick", "second", "alive_ct", "alive_t", "ct_alive_minus_t", "bomb_planted", "seconds_since_round_live",
            "seconds_since_plant", "label_ct_win", "w_round", "era", "map_name", "score_lead_diag"]
    G = W.load_grid(store, cols=cols)
    G = G.with_columns([(pl.col("map_name") == f"de_{m}").cast(pl.Float32).alias(f"map_is_{m}") for m in T.MAPS])
    G = W.drop_roster_defect_rounds(G, store)
    feats = list(T.BASIC_FEATS) + ["score_lead_diag"]
    mono = list(W.mono_for(feats))
    print(f"monotone for score lead: {mono[-1]}; rows {G.height:,} ({time.time() - t0:.0f}s)", flush=True)

    sp = W.make_split("random", G, seed=0)
    tr, va, te = sorted(sp["train"]), sorted(sp["val"]), sorted(sp["test"])
    _, Xtr, ytr = W.arrays(G, tr, feats, stride=1)
    sub_va, Xva, yva = W.arrays(G, va, feats, stride=1)
    sub_te, Xte, yte = W.arrays(G, te, feats, stride=1)
    print(f"train {len(ytr):,}, validation {len(yva):,}, test {len(yte):,} ({time.time() - t0:.0f}s)", flush=True)

    m, best = W.fit_xgb(Xtr, ytr, Xva, yva, mono, seed=0, device="cpu")
    p = W.predict(m, best, Xte)
    pva = W.predict(m, best, Xva)
    print(f"fitted: best iteration {best} ({time.time() - t0:.0f}s)", flush=True)

    # the served 13-input model on the same rows (its inputs are the first 13 columns)
    ref_meta = json.load(open(os.path.join(W.ROOT, "models", "winprob_v4_basic.meta.json"), encoding="utf-8"))
    ref = xgb.Booster()
    ref.load_model(os.path.join(W.ROOT, "models", "winprob_v4_basic.json"))
    n13 = len(T.BASIC_FEATS)
    p_ref = ref.predict(xgb.DMatrix(Xte[:, :n13]), iteration_range=(0, int(ref_meta["best_iteration"]) + 1))
    pva_ref = ref.predict(xgb.DMatrix(Xva[:, :n13]), iteration_range=(0, int(ref_meta["best_iteration"]) + 1))

    ev = W.evaluate(sub_te, yte, p, ref=p_ref)
    evv = W.evaluate(sub_va, yva, pva, ref=pva_ref)
    out = {
        "inputs": len(feats), "best_iteration": best, "monotone_score_lead": mono[-1],
        "train_rows": int(len(ytr)), "validation_rows": int(len(yva)), "test_rows": int(len(yte)),
        "test_matches": ev["n_matches"], "validation_matches": evv["n_matches"],
        "test_logloss": ev["logloss"], "test_logloss_ci95": ev["logloss_ci95"],
        "test_delta_vs_served_13": ev["delta_vs_ref"], "test_delta_vs_served_13_ci95": ev["delta_vs_ref_ci95"],
        "validation_logloss": evv["logloss"], "validation_delta_vs_served_13": evv["delta_vs_ref"],
        "validation_delta_vs_served_13_ci95": evv["delta_vs_ref_ci95"],
        "served_13_test_logloss_reported": ref_meta["test_logloss"], "served_13_validation_logloss_reported": ref_meta["validation_logloss"],
    }
    W.dump_json(out, os.path.join(W.ROOT, "data", store, "exp_scorelead.json"))
    print(f"test {ev['logloss']:.5f}; delta vs served {ev['delta_vs_ref']:+.5f} [{ev['delta_vs_ref_ci95'][0]:+.5f}, {ev['delta_vs_ref_ci95'][1]:+.5f}]; "
          f"validation delta {evv['delta_vs_ref']:+.5f} [{evv['delta_vs_ref_ci95'][0]:+.5f}, {evv['delta_vs_ref_ci95'][1]:+.5f}] "
          f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
