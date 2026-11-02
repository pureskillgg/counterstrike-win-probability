"""Learning curve in the units a reader can use: how often the favoured side wins, against the number of training matches.   (CUDA XGBoost)

  ...tabicl\\python.exe -I -X utf8 scripts\\model\\exp_curve_accuracy.py [--stride 3]

Nested random subsets (5, 10, 20, 40, 70, 100 percent) of the frozen-split TRAINING matches; same settings as the final model (59 inputs, depth 4); validation matches for early
stopping; every model scored on the same frozen TEST matches (every second). Per point: accuracy (the side with probability above 0.5 is the side that wins the round, counted over
seconds), AUC, log loss, each with a match-cluster 95% interval (bootstrap over test matches). Writes data/<store>/exp_curve_accuracy.json.
"""
import os

os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
import argparse
import json
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import polars as pl

import wp_common as W

sys.stdout.reconfigure(encoding="utf-8")


def cluster_ci(mk_codes, num, den, n_boot=1000, seed=0):
    """ratio-of-sums bootstrap over matches: sum(num) / sum(den) with whole matches resampled."""
    k = int(mk_codes.max()) + 1
    s = np.bincount(mk_codes, weights=num, minlength=k)
    c = np.bincount(mk_codes, weights=den, minlength=k)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(n_boot, k))
    r = s[idx].sum(1) / c[idx].sum(1)
    return float(s.sum() / c.sum()), [float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="combined")
    ap.add_argument("--stride", type=int, default=3)
    a = ap.parse_args()
    d = os.path.join(W.ROOT, "data", a.store)
    t0 = time.time()
    feats = json.load(open(os.path.join(W.ROOT, "models", "winprob_v3.meta.json"), encoding="utf-8"))["features"]
    need = list(dict.fromkeys(["match_key", "round", "tick", "second", "label_ct_win", "w_round"] + feats))
    lf = pl.scan_parquet(os.path.join(d, "cache_snapshot_2026.parquet")).select(need)
    kdf = lf.select("match_key").unique().collect().with_columns(pl.lit(1.0).alias("w_round"), pl.lit(0).alias("label_ct_win"), pl.lit("2026").alias("era"), pl.lit("").alias("match_date"))
    sp = W.make_split("random", kdf, seed=0)
    test_keys = sorted(sp["test"])
    G = lf.filter((pl.col("second") % a.stride == 0) | pl.col("match_key").is_in(test_keys)).collect()
    print(f"rows kept {G.height:,}; train {len(sp['train'])} val {len(sp['val'])} test {len(sp['test'])} matches ({time.time() - t0:.0f}s)", flush=True)
    sub, Xte, yte = W.arrays(G, sp["test"], feats)
    codes = pl.Series(sub["match_key"]).cast(pl.Categorical).to_physical().to_numpy().astype(np.int64)
    tr_sorted = sorted(sp["train"])
    order = np.random.default_rng(0).permutation(len(tr_sorted))
    out = {"stride": a.stride, "test_matches": len(sp["test"]), "test_seconds": int(len(yte)), "points": []}
    for frac in (0.05, 0.10, 0.20, 0.40, 0.70, 1.00):
        t1 = time.time()
        n = max(50, int(round(frac * len(tr_sorted))))
        keys = {tr_sorted[i] for i in order[:n]}
        _, Xtr, ytr = W.arrays(G, keys, feats, stride=a.stride)
        _, Xva, yva = W.arrays(G, sp["val"], feats, stride=a.stride)
        m, best = W.fit_xgb(Xtr, ytr, Xva, yva, W.mono_for(feats), seed=0, device="cuda")
        p = W.predict(m, best, Xte)
        r = W.evaluate(sub, yte, p.astype(np.float64))
        correct = ((p > 0.5).astype(np.int64) == yte).astype(np.float64)
        acc, acc_ci = cluster_ci(codes, correct, np.ones_like(correct))
        row = {"fraction": frac, "train_matches": n, "accuracy": acc, "accuracy_ci95": acc_ci, "auc": r["auc"], "logloss": r["logloss"], "logloss_ci95": r["logloss_ci95"], "best_iteration": int(best)}
        out["points"].append(row)
        print(f"{n} matches: accuracy {acc:.4f} {[round(x, 4) for x in acc_ci]}  AUC {r['auc']:.4f}  log loss {r['logloss']:.4f} ({time.time() - t1:.0f}s)", flush=True)
    W.dump_json(out, os.path.join(d, "exp_curve_accuracy.json"))
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
