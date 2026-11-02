"""Sensitivity to the roster-defect drop. The served 13-input model is trained and scored without the rounds that have a missing player (about
0.4% of rounds). This script trains the same model with those rounds included, then compares the two on the same clean test seconds (the
served model's test rows), with a paired match-cluster interval. It also scores the served model on the defect rounds, so the two models can
be compared where they differ.

Settings as winprob_v4_basic: depth 4, eta 0.05, min child weight 20, subsample 0.8, early stopping after 100 rounds on the clean validation
seconds, seed 0, monotone alive counts. CPU (xgboost hist).

  ...python.exe -I -X utf8 scripts\\model\\exp_roster_sensitivity.py
Writes data/combined/exp_roster_sensitivity.json.
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
    d = os.path.join(W.ROOT, "data", store)
    cols = ["match_key", "round", "tick", "second", "alive_ct", "alive_t", "ct_alive_minus_t", "bomb_planted", "seconds_since_round_live",
            "seconds_since_plant", "label_ct_win", "w_round", "era", "map_name"]
    G_all = W.load_grid(store, cols=cols)
    G_all = G_all.with_columns([(pl.col("map_name") == f"de_{m}").cast(pl.Float32).alias(f"map_is_{m}") for m in T.MAPS])
    G_clean = W.drop_roster_defect_rounds(G_all, store)
    feats = list(T.BASIC_FEATS)
    mono = list(W.mono_for(feats))
    clean_rounds = G_clean.select(["match_key", "round"]).unique()
    defect = G_all.join(clean_rounds.with_columns(pl.lit(True).alias("clean")), on=["match_key", "round"], how="left").filter(pl.col("clean").is_null())
    defect_rounds = defect.select(["match_key", "round"]).unique()
    print(f"rounds total {G_all.select(['match_key','round']).unique().height:,}; defect rounds {defect_rounds.height:,}; "
          f"defect rows {defect.height:,} ({time.time() - t0:.0f}s)", flush=True)

    # the split, as train_final.py makes it (on the clean frame, so the matches are the served model's)
    sp = W.make_split("random", G_clean, seed=0)
    tr, va, te = sorted(sp["train"]), sorted(sp["val"]), sorted(sp["test"])
    _, Xtr_c, ytr_c = W.arrays(G_clean, tr, feats, stride=1)
    sub_va, Xva, yva = W.arrays(G_clean, va, feats, stride=1)
    sub_te, Xte, yte = W.arrays(G_clean, te, feats, stride=1)
    # the same training matches with every round (defect rounds included)
    sub_tr_all, Xtr_all, ytr_all = W.arrays(G_all, tr, feats, stride=1)
    print(f"clean train {len(ytr_c):,} seconds; all-round train {len(ytr_all):,}; validation {len(yva):,}; test {len(yte):,} ({time.time() - t0:.0f}s)",
          flush=True)

    # the served model on the clean test seconds
    served = xgb.Booster()
    served.load_model(os.path.join(W.ROOT, "models", "winprob_v4_basic.json"))
    best_served = int(json.load(open(os.path.join(W.ROOT, "models", "winprob_v4_basic.meta.json"), encoding="utf-8"))["best_iteration"])
    p_served = served.predict(xgb.DMatrix(Xte), iteration_range=(0, best_served + 1))

    # the same model trained with the defect rounds included
    m_all, best_all = W.fit_xgb(Xtr_all, ytr_all, Xva, yva, mono, seed=0, device="cpu")
    p_all = W.predict(m_all, best_all, Xte)
    print(f"all-round model: best iteration {best_all} ({time.time() - t0:.0f}s)", flush=True)

    ev = W.evaluate(sub_te, yte, p_all, ref=p_served)
    out = {
        "defect_rounds": int(defect_rounds.height), "defect_rows": int(defect.height),
        "clean_train_seconds": int(len(ytr_c)), "all_round_train_seconds": int(len(ytr_all)),
        "served_best_iteration": best_served, "all_round_best_iteration": int(best_all),
        "test_seconds": int(len(yte)), "test_matches": ev["n_matches"],
        "test_logloss_served_clean": W.evaluate(sub_te, yte, p_served)["logloss"],
        "test_logloss_all_round_clean": ev["logloss"],
        "all_round_minus_served_clean_test": ev["delta_vs_ref"], "ci95": ev["delta_vs_ref_ci95"],
    }

    W.dump_json(out, os.path.join(d, "exp_roster_sensitivity.json"))
    print(json.dumps({k: out[k] for k in ["defect_rounds", "all_round_minus_served_clean_test", "ci95"]}, indent=1), flush=True)
    print(f"done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
