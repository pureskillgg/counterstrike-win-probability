"""Paired log-loss comparison of two retrains on the same frozen test rows, overall and by situation.

Situations follow the intern grader's time thirds: before the plant by seconds into the round (under 38.33 lot, under 76.67 medium,
else little), after the plant by seconds since the plant (under 13.33 lot, under 26.67 medium, else little). Intervals resample
test matches, as the grader does.

Difference = log loss of A minus log loss of B, so a positive number means B is better.

    python analysis/paired_retrain.py --a <preds_a.parquet> --b <preds_b.parquet> --tag control_vs_expanded
"""
import argparse
import os

import numpy as np
import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "G:/pureskill/csds-cs2"
KEY = ["match_key", "round", "second"]


def ll(p, y):
    p = np.clip(np.asarray(p, dtype="float64"), 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def main():
    ap = argparse.ArgumentParser(description="Paired log-loss comparison of two retrains on the frozen test rows")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--tag", default="paired")
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "docs", "retrain_results"))
    a = ap.parse_args()

    pa = pl.read_parquet(a.a).select(KEY + ["label_ct_win", "p_base"]).rename({"p_base": "pa"})
    pb = pl.read_parquet(a.b).select(KEY + ["p_base"]).rename({"p_base": "pb"})
    cache = (pl.scan_parquet(ROOT + "/data/combined/cache_snapshot_2026.parquet")
             .select(KEY + ["bomb_planted", "seconds_since_plant"]).with_columns(pl.col("second").cast(pl.Int64)).collect())
    df = pa.join(pb, on=KEY, how="inner").with_columns(pl.col("second").cast(pl.Int64)).join(cache, on=KEY, how="left")
    print("rows", df.height, "matches", df["match_key"].n_unique(), "missing planted", int(df["bomb_planted"].is_null().sum()))

    df = df.with_columns(
        pl.when(pl.col("bomb_planted") == 1).then(pl.lit("post-plant")).otherwise(pl.lit("pre-plant")).alias("phase"),
        pl.when(pl.col("bomb_planted") == 1)
        .then(pl.when(pl.col("seconds_since_plant") < 40 / 3).then(pl.lit("lot"))
              .when(pl.col("seconds_since_plant") < 80 / 3).then(pl.lit("medium")).otherwise(pl.lit("little")))
        .otherwise(pl.when(pl.col("second") < 115 / 3).then(pl.lit("lot"))
                   .when(pl.col("second") < 230 / 3).then(pl.lit("medium")).otherwise(pl.lit("little"))).alias("tb"),
    ).with_columns(pl.concat_str([pl.col("phase"), pl.lit(" "), pl.col("tb"), pl.lit(" left")]).alias("group"))

    y = df["label_ct_win"].to_numpy().astype("float64")
    d = ll(df["pa"].to_numpy(), y) - ll(df["pb"].to_numpy(), y)
    mids = df["match_key"].to_numpy()
    ids, inv = np.unique(mids, return_inverse=True)
    m = len(ids)
    rng = np.random.default_rng(a.seed)
    picks = rng.integers(0, m, (a.reps, m))

    def stat(mask):
        sums = np.bincount(inv[mask], weights=d[mask], minlength=m)
        cnt = np.bincount(inv[mask], minlength=m).astype("float64")
        est = sums.sum() / cnt.sum()
        boot = sums[picks].sum(axis=1) / np.maximum(cnt[picks].sum(axis=1), 1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        return est, lo, hi, int(mask.sum()), int((cnt > 0).sum())

    rows = []
    est, lo, hi, n, nm = stat(np.ones(len(d), dtype=bool))
    rows.append({"group": "overall", "datapoints": n, "matches": nm, "diff": est, "ci_low": lo, "ci_high": hi})
    g = df["group"].to_numpy()
    for name in sorted(set(g)):
        est, lo, hi, n, nm = stat(g == name)
        rows.append({"group": name, "datapoints": n, "matches": nm, "diff": est, "ci_low": lo, "ci_high": hi})
    res = pl.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"paired_{a.tag}.csv")
    res.write_csv(path)
    print(f"difference = LL(A) - LL(B); positive = B better; written {path}")
    for r in res.iter_rows(named=True):
        print(f"  {r['group']:<26} datapoints {r['datapoints']:>9,}  matches {r['matches']:>4}  "
              f"{r['diff']:+.5f} [{r['ci_low']:+.5f}, {r['ci_high']:+.5f}]")


if __name__ == "__main__":
    main()
