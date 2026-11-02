"""Submission numbers for the 57-input model (59 minus heavy group, no score lead): winprob_v6_drop_heavy.

Scores the saved booster on the frozen split (validation 1,168 matches, test 3,059 matches on the 2026 cache rows), with match-cluster 95%
intervals (1,000 resamples, seed 0), overall and by round group. Also the paired difference against the 58-input production ensemble
(score lead, five seeds) on the same test seconds, from the saved test predictions.

  ...python.exe -I -X utf8 scripts\\model\\exp_submission_57.py
Writes data/combined/exp_submission_57.json.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import polars as pl
import xgboost as xgb

import wp_common as W

sys.stdout.reconfigure(encoding="utf-8")
N_BOOT = 1000


def logloss(y, p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def match_bootstrap(keys, d, n=N_BOOT, seed=0):
    _, inv = np.unique(np.asarray(keys), return_inverse=True)
    s = np.bincount(inv, weights=np.asarray(d, dtype=np.float64))
    c = np.bincount(inv).astype(np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(n, len(s)))
    means = s[idx].sum(axis=1) / c[idx].sum(axis=1)
    return float(s.sum() / c.sum()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    d = os.path.join(W.ROOT, "data", "combined")
    meta = json.load(open(os.path.join(W.ROOT, "models", "winprob_v6_drop_heavy.meta.json"), encoding="utf-8"))
    feats = meta["features"]
    assert len(feats) == 57 and not any("heavy" in f or f == "score_lead_diag" for f in feats)
    bst = xgb.Booster()
    bst.load_model(os.path.join(W.ROOT, "models", "winprob_v6_drop_heavy.json"))
    best = int(meta["best_iteration"])

    man = json.load(open(os.path.join(d, "split_manifest.json"), encoding="utf-8"))["assign"]
    cols = ["match_key", "round", "tick", "label_ct_win"] + feats
    cache = pl.read_parquet(os.path.join(d, "cache_snapshot_2026.parquet"), columns=cols)
    cache = cache.filter(pl.col("match_key").str.starts_with("csds/2026/"))
    cache = cache.with_columns(pl.col("match_key").map_elements(lambda k: man.get(k, ""), return_dtype=pl.Utf8).alias("split"))

    out = {"model": "winprob_v6_drop_heavy (57 inputs, no score lead, single seed)", "best_iteration": best, "splits": {}}
    for name, tag in [("validation", "val"), ("test", "test")]:
        sub = cache.filter(pl.col("split") == tag)
        X = sub.select(feats).to_numpy().astype(np.float32)
        y = sub["label_ct_win"].to_numpy().astype(np.float64)
        p = bst.predict(xgb.DMatrix(X, feature_names=feats), iteration_range=(0, best + 1)).astype(np.float64)
        keys = sub["match_key"].to_numpy()
        rnd = sub["round"].to_numpy()
        ll = logloss(y, p)
        mean, lo, hi = match_bootstrap(keys, ll)
        entry = {"matches": int(len(np.unique(keys))), "seconds": int(sub.height), "logloss": mean, "ci95": [lo, hi], "groups": {}}
        for gname, gm in [("pistol (1, 13)", np.isin(rnd, [1, 13])), ("after pistol (2, 14)", np.isin(rnd, [2, 14])),
                          ("other rounds", ~np.isin(rnd, [1, 2, 13, 14]))]:
            gmean, glo, ghi = match_bootstrap(keys[gm], ll[gm])
            entry["groups"][gname] = {"seconds": int(gm.sum()), "logloss": gmean, "ci95": [glo, ghi]}
        out["splits"][name] = entry
        print(f"{name}: {entry['matches']} matches, {entry['seconds']:,} seconds: {mean:.6f} [{lo:.6f}, {hi:.6f}]", flush=True)
        for gname, g in entry["groups"].items():
            print(f"   {gname}: {g['seconds']:,} seconds: {g['logloss']:.6f} [{g['ci95'][0]:.6f}, {g['ci95'][1]:.6f}]", flush=True)
        if name == "test":
            test_keys, test_y, test_p57 = keys, y, p

    # paired difference against the 58-input production ensemble (score lead, five seeds) on the same test seconds
    ex = pl.read_parquet(os.path.join(d, "export_winprob_v8_ensemble_test_predictions.parquet"))
    sub = cache.filter(pl.col("split") == "test").select(["match_key", "round", "tick", "label_ct_win"]).with_columns(pl.Series("p57", test_p57))
    both = sub.join(ex, on=["match_key", "round", "tick"], how="inner")
    assert both.height == sub.height == ex.height, (both.height, sub.height, ex.height)
    y58 = both["label_ct_win"].to_numpy().astype(np.float64)
    p57 = both["p57"].to_numpy().astype(np.float64)
    p58 = both["p"].to_numpy().astype(np.float64)
    d_loss = logloss(y58, p57) - logloss(y58, p58)
    mean, lo, hi = match_bootstrap(both["match_key"].to_numpy(), d_loss)
    out["paired_57_minus_58_ensemble_test"] = {"difference": mean, "ci95": [lo, hi], "logloss_58_ensemble": float(logloss(y58, p58).mean()),
                                               "logloss_57": float(logloss(y58, p57).mean())}
    print(f"57 minus 58 ensemble (test, paired): {mean:+.6f} [{lo:+.6f}, {hi:+.6f}]", flush=True)
    W.dump_json(out, os.path.join(d, "exp_submission_57.json"))


if __name__ == "__main__":
    main()
