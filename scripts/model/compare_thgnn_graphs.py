"""Paired comparison of two THGNN-lite runs on the same de_mirage test seconds, for three seeds: the navigation-region graph against the k-means graph.

Seed 0 reads the original prediction files; seeds 1 and 2 read the _seed{s} files. Within each seed, both runs share the extended round-state table, cache and budget (8,544 steps, 6 epochs), so their prediction files share row order. The script checks that, joins the round winner, and reports for each seed the paired match-cluster bootstrap (1,000 resamples, seed 0) of log loss: navgraph minus kmeans.

Pre-registered (written before the three-seed run): the navigation-region graph is better if the mean over seeds 0, 1 and 2 of the paired difference is <= -0.0005 AND every seed's 95% CI upper bound is < 0. The seed-0 difference from the earlier single-run comparison (+0.00037, paired_thgnn_navgraph_vs_kmeans.json) must reproduce; it checks this script and is not a new test.

Rows: the trainer's usable test seconds (865,974 matches the metrics files; 687 matches). Needs torch, so run in the torch environment:
  G:/pureskill/csds-cs2/.pixi/envs/torch/python.exe scripts/model/compare_thgnn_graphs.py
Writes data/temporal/paired_thgnn_navgraph_vs_kmeans_3seed.json.
"""
import json
import os
import sys

import numpy as np
import polars as pl

ROOT = "G:/pureskill/csds-cs2"
T = os.path.join(ROOT, "data", "temporal")
OUT = os.path.join(T, "paired_thgnn_navgraph_vs_kmeans_3seed.json")
SEEDS = [0, 1, 2]


def pred_paths(seed):
    suffix = "" if seed == 0 else f"_seed{seed}"
    return (os.path.join(T, f"de_mirage_ext_navgraph_ms8544{suffix}_pred.npz"),
            os.path.join(T, f"de_mirage_ext_thgnn_lite_ms8544{suffix}_pred.npz"))


def row_loss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def usable_mask():
    """The trainer's usable rows (a winner and a snapshot state), over the full round-state table. Same set as the metrics files."""
    sys.path.insert(0, os.path.join(ROOT, "scripts", "model"))
    import train_temporal_round_model as ttr  # noqa: E402  (needs torch: run in the torch environment)
    _, _, _, _, _, usable = ttr.build_table("de_mirage", "de_mirage_ext", "cache_snapshot_2026_ext.parquet")
    return np.asarray(usable, dtype=bool)


def load(path, usable):
    z = np.load(path)
    t = (z["split"] == 2) & usable
    keys = z["match_keys"][z["match_idx"][t]]
    return {"match_key": keys, "round": z["round"][t].astype(np.int64), "second": z["second"][t].astype(np.int64), "p": z["p"][t].astype(np.float64)}


def compare(seed, rounds, usable):
    a_path, b_path = pred_paths(seed)
    a, b = load(a_path, usable), load(b_path, usable)
    for k in ("match_key", "round", "second"):
        if not np.array_equal(a[k], b[k]):
            raise SystemExit(f"seed {seed}: test rows differ in {k}; the two runs are not aligned")
    df = pl.DataFrame({"match_key": a["match_key"], "round": a["round"], "second": a["second"], "pa": a["p"], "pb": b["p"]})
    df = df.join(rounds, on=["match_key", "round"], how="inner")
    y = (df["final"].to_numpy() == 3).astype(np.float64)
    la, lb = row_loss(y, df["pa"].to_numpy()), row_loss(y, df["pb"].to_numpy())
    diff = la - lb
    match = df["match_key"].to_numpy()
    uniq, inv = np.unique(match, return_inverse=True)
    sums = np.zeros(uniq.size)
    counts = np.zeros(uniq.size)
    np.add.at(sums, inv, diff)
    np.add.at(counts, inv, 1)
    rng = np.random.default_rng(0)
    boot = np.empty(1000)
    for i in range(1000):
        pick = rng.integers(0, uniq.size, uniq.size)
        boot[i] = sums[pick].sum() / counts[pick].sum()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"seed": seed, "rows": int(df.height), "matches": int(uniq.size),
            "log_loss_navgraph": float(la.mean()), "log_loss_kmeans": float(lb.mean()),
            "difference_navgraph_minus_kmeans": float(diff.mean()), "ci95_match_bootstrap": [float(lo), float(hi)],
            "ci_upper_below_zero": bool(hi < 0)}


def main():
    usable = usable_mask()
    rounds = (pl.read_parquet(os.path.join(ROOT, "data", "combined", "rounds.parquet"), columns=["match_key", "round", "final"])
              .with_columns(pl.col("round").cast(pl.Int64)).unique(["match_key", "round"], keep="first"))
    per_seed = [compare(s, rounds, usable) for s in SEEDS]
    mean = float(np.mean([r["difference_navgraph_minus_kmeans"] for r in per_seed]))
    res = {"per_seed": per_seed, "mean_difference": mean,
           "passes": bool(mean <= -0.0005 and all(r["ci_upper_below_zero"] for r in per_seed))}
    os.makedirs(T, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
