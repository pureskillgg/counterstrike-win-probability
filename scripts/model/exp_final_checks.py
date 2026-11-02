"""Two last checks before the package is closed.

(1) Leakage canary. A deliberately leaking feature (the alive count five seconds in the future, within the round) is added to the 59-input set
    and trained like the real model. If the evaluation works, this canary should look much better than any real feature. It is a control, not a
    model: it shows the metric would catch future information.
(2) Weighted random draws. The n = 64 per-round sweep (random_tick_pool59_s0) against the same draws with each draw weighted by its round's
    length divided by 64, so each round counts as much as its seconds do in the every-second data. Same 59 inputs, same settings, seed 0,
    validation for early stopping, test scored once against the every-second rows. If the gap closes, round weighting explains it.

  ...python.exe -I -X utf8 scripts\\model\\exp_final_checks.py
Writes data/combined/exp_final_checks.json.
"""
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


def fit(X, y, Xva, yva, mono, weight=None):
    clf = xgb.XGBClassifier(n_estimators=3000, learning_rate=0.05, max_depth=4, min_child_weight=20, subsample=0.8, tree_method="hist",
                            device="cuda", early_stopping_rounds=100, eval_metric="logloss", random_state=0,
                            monotone_constraints=tuple(mono))
    clf.fit(X, y, sample_weight=weight, eval_set=[(Xva, yva)], verbose=False)
    return clf, int(clf.best_iteration)


def main():
    t0 = time.time()
    d = os.path.join(W.ROOT, "data", "combined")
    ref_meta = json.load(open(os.path.join(W.ROOT, "models", "winprob_v4.meta.json"), encoding="utf-8"))
    feats59 = json.load(open(os.path.join(W.ROOT, "models", "winprob_v3.meta.json"), encoding="utf-8"))["features"]
    mono59 = [int(ref_meta["monotone_constraints"][f]) for f in feats59]
    need = list(dict.fromkeys(["match_key", "round", "tick", "second", "label_ct_win", "w_round"] + feats59))
    G = pl.scan_parquet(os.path.join(d, "cache_snapshot_2026.parquet")).select(need).with_columns(pl.col(feats59).cast(pl.Float32)).collect()
    kdf = G.select("match_key").unique().with_columns(pl.lit(1.0).alias("w_round"), pl.lit(0).alias("label_ct_win"),
                                                      pl.lit("2026").alias("era"), pl.lit("").alias("match_date"))
    sp = W.make_split("random", kdf, seed=0)
    tr, va, te = sorted(sp["train"]), sorted(sp["val"]), sorted(sp["test"])
    del kdf
    out = {}

    # (1) canary: alive_ct five seconds ahead within the round (the cache rows are one per second, sorted by tick within a round)
    G = G.sort(["match_key", "round", "tick"]).with_columns(pl.col("alive_ct").shift(-5).over(["match_key", "round"]).fill_null(0).cast(pl.Float32)
                                                           .alias("canary_future_alive_ct"))
    sub_va, Xva, yva = W.arrays(G, va, feats59 + ["canary_future_alive_ct"], stride=1)
    sub_te, Xte, yte = W.arrays(G, te, feats59 + ["canary_future_alive_ct"], stride=1)
    sub_tr, Xtr, ytr = W.arrays(G, tr, feats59 + ["canary_future_alive_ct"], stride=3)
    clf, best = fit(Xtr, ytr, Xva, yva, mono59 + [0])
    p = clf.predict_proba(Xte, iteration_range=(0, best + 1))[:, 1]
    canary = W.evaluate(sub_te, yte, p)["logloss"]
    out["canary"] = {"test_logloss_with_future_feature": canary, "best_iteration": best, "note": "stride-3 training, 59 inputs plus the future alive count"}
    print(f"canary (59 inputs + alive count 5 s in the future): test {canary:.6f} ({time.time() - t0:.0f}s)", flush=True)
    W.dump_json(out, os.path.join(d, "exp_final_checks.json"))
    del Xtr, ytr, Xva, Xte, sub_va, sub_te, sub_tr

    # (2) weighted random draws: 64 per round (the pool), unweighted vs weighted by round length / 64
    rounds_len = G.group_by(["match_key", "round"]).agg(pl.len().alias("L"))
    sub_va, Xva, yva = W.arrays(G, va, feats59, stride=1)
    sub_te, Xte, yte = W.arrays(G, te, feats59, stride=1)
    pool = pl.read_parquet(os.path.join(d, "random_tick_pool59_s0.parquet"), columns=["match_key", "round", "draw_rank", "label_ct_win"] + feats59)
    pool = pool.filter(pl.col("match_key").is_in(tr)).join(rounds_len, on=["match_key", "round"], how="inner")
    pool = pool.with_columns((pl.col("L") / pl.len().over(["match_key", "round"])).alias("w"))
    print(f"pool rows {pool.height:,} over {pool.select(['match_key', 'round']).unique().height:,} rounds ({time.time() - t0:.0f}s)", flush=True)
    Xp = pool.select(feats59).to_numpy().astype("float32")
    yp = pool["label_ct_win"].to_numpy()
    wp = pool["w"].to_numpy().astype("float32")
    clf0, b0 = fit(Xp, yp, Xva, yva, mono59)
    p0 = clf0.predict_proba(Xte, iteration_range=(0, b0 + 1))[:, 1]
    ev0 = W.evaluate(sub_te, yte, p0)
    clf1, b1 = fit(Xp, yp, Xva, yva, mono59, weight=wp)
    p1 = clf1.predict_proba(Xte, iteration_range=(0, b1 + 1))[:, 1]
    ev1 = W.evaluate(sub_te, yte, p1, ref=p0)
    out["weighted_draws"] = {"unweighted_test_logloss": ev0["logloss"], "weighted_test_logloss": ev1["logloss"],
                             "weighted_minus_unweighted": ev1["delta_vs_ref"], "ci95": ev1["delta_vs_ref_ci95"],
                             "best_iterations": [b0, b1]}
    lo, hi = ev1["delta_vs_ref_ci95"]
    print(f"n=64 unweighted: test {ev0['logloss']:.6f}; weighted by round length / 64: test {ev1['logloss']:.6f} "
          f"(difference {ev1['delta_vs_ref']:+.5f} [{lo:+.5f}, {hi:+.5f}]) ({time.time() - t0:.0f}s)", flush=True)
    W.dump_json(out, os.path.join(d, "exp_final_checks.json"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
