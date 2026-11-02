"""Final model: the 58-input score-lead set (59 minus heavy, plus score lead), averaged over the five saved XGBoost seeds.

Re-derives validation and test log loss from the saved boosters (models/winprob_v8_seed0..4.json) on the frozen split, so the reported
numbers come from the files, not from the run that trained them. The average is the mean of the five probabilities, and the metadata lists
each seed's best iteration. Rules: frozen split (data/combined/split_manifest.json), validation for early stopping and selection, test
scored once, baseline XGBoost settings (depth 4, eta 0.05, min child weight 20, subsample 0.8), monotone alive counts, score lead unconstrained.

Writes models/winprob_v8_ensemble.meta.json (served: false).

  ...python.exe -I -X utf8 scripts\\model\\finalize_ensemble58.py
"""
import hashlib
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
SEEDS = [0, 1, 2, 3, 4]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main():
    t0 = time.time()
    d = os.path.join(W.ROOT, "data", "combined")
    v7 = json.load(open(os.path.join(W.ROOT, "models", "winprob_v7_59_drop_heavy_scorelead.meta.json"), encoding="utf-8"))
    feats = v7["features"]
    mono = [int(v7["monotone_constraints"][f]) for f in feats]
    seed_rows = json.load(open(os.path.join(d, "exp_seed_avg58.json"), encoding="utf-8"))["seeds"]
    best = {r["seed"]: r["best_iteration"] for r in seed_rows}

    feats59 = [f for f in feats if f != "score_lead_diag"]  # the cache holds the 59 snapshot inputs; score lead comes from the grid
    need = list(dict.fromkeys(["match_key", "round", "tick", "second", "label_ct_win", "w_round"] + feats59))
    G = pl.scan_parquet(os.path.join(d, "cache_snapshot_2026.parquet")).select(need).collect()
    G = G.with_columns(pl.col(feats59).cast(pl.Float32))
    lead = (pl.scan_parquet(os.path.join(d, "grid.parquet")).filter(pl.col("era") == "2026").select(["match_key", "round", "score_lead_diag"])
            .unique(subset=["match_key", "round"]).collect())
    G = G.join(lead, on=["match_key", "round"], how="left").with_columns(pl.col("score_lead_diag").fill_null(0).cast(pl.Float32))
    kdf = G.select("match_key").unique().with_columns(pl.lit(1.0).alias("w_round"), pl.lit(0).alias("label_ct_win"),
                                                      pl.lit("2026").alias("era"), pl.lit("").alias("match_date"))
    sp = W.make_split("random", kdf, seed=0)
    tr, va, te = sorted(sp["train"]), sorted(sp["val"]), sorted(sp["test"])
    del kdf
    sub_va, Xva, yva = W.arrays(G, va, feats, stride=1)
    sub_te, Xte, yte = W.arrays(G, te, feats, stride=1)
    dva, dte = xgb.DMatrix(Xva), xgb.DMatrix(Xte)

    per_seed = []
    pva_all, pte_all = [], []
    for seed in SEEDS:
        b = xgb.Booster()
        b.load_model(os.path.join(W.ROOT, "models", f"winprob_v8_seed{seed}.json"))
        pva = b.predict(dva, iteration_range=(0, best[seed] + 1))
        pte = b.predict(dte, iteration_range=(0, best[seed] + 1))
        pva_all.append(pva)
        pte_all.append(pte)
        ll_va = W.evaluate(sub_va, yva, pva)["logloss"]
        ll_te = W.evaluate(sub_te, yte, pte)["logloss"]
        per_seed.append({"seed": seed, "best_iteration": best[seed], "validation_logloss": ll_va, "test_logloss": ll_te})
        print(f"seed {seed}: validation {ll_va:.6f}; test {ll_te:.6f} ({time.time() - t0:.0f}s)", flush=True)

    pva_mean, pte_mean = np.mean(pva_all, axis=0), np.mean(pte_all, axis=0)
    ll_va = W.evaluate(sub_va, yva, pva_mean)["logloss"]
    ev_te = W.evaluate(sub_te, yte, pte_mean, ref=pte_all[0])
    ll_te = ev_te["logloss"]
    lo, hi = ev_te["delta_vs_ref_ci95"]
    print(f"ensemble of 5 seeds: validation {ll_va:.6f}; test {ll_te:.6f} (rows {len(yte):,}); vs seed 0 {ev_te['delta_vs_ref']:+.5f} "
          f"[{lo:+.5f}, {hi:+.5f}] ({time.time() - t0:.0f}s)", flush=True)

    meta = {"version": "winprob_v8_ensemble", "features": feats, "monotone_constraints": dict(zip(feats, mono)),
            "members": per_seed, "ensemble": "mean of the five seed probabilities", "split": "random",
            "split_manifest_sha256": sha(os.path.join(d, "split_manifest.json")), "drop_roster_defect_rounds": True,
            "params": {"depth": 4, "eta": 0.05, "mcw": 20, "subsample": 0.8, "early_stopping_rounds": 100},
            "validation_logloss": ll_va, "test_logloss": ll_te, "test_rows": int(len(yte)),
            "test_delta_vs_seed0": ev_te["delta_vs_ref"], "test_delta_vs_seed0_ci95": [lo, hi],
            "selection": "validation log loss (settings chosen on validation; test scored once)",
            "served": False, "note": "58 inputs (59 minus heavy, plus score lead), needs per-tick player status to serve"}
    with open(os.path.join(W.ROOT, "models", "winprob_v8_ensemble.meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote models/winprob_v8_ensemble.meta.json ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
