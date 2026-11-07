"""M4 bound: how much log loss can a per-map (or per-map x site) constant prior buy?

Round level, 2026 kept rounds (28,683). Protocol: fit the per-group CT win rate on training-split
rounds (primary split, the bake-off's own split) and score test-split rounds. Compared with one global
constant. This is an upper bound on what a map or site prior can add with no game-state features,
because the model also knows alive counts, time and score. Also reports the in-sample gain.
Post-plant only for the site version (planted rounds, cluster from training planted rounds).
"""
import math
import os

import numpy as np
import polars as pl

SCR = (r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
       r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\M4")
BASE_LL = 0.43660  # bake-off primary test log loss (reproduced in M4_models)


def ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def fit_rate(df, key, prior_n=0.0, global_rate=None):
    g = df.group_by(key).agg(pl.col("ct_win").sum().alias("k"), pl.len().alias("n"))
    rates = {}
    for r in g.iter_rows(named=True):
        n = r["n"]
        k = r["k"]
        # additive shrinkage toward the global rate; prior_n = 0 means raw rate
        rates[r[key] if isinstance(key, str) else tuple(r[c] for c in key)] = (k + prior_n * global_rate) / (n + prior_n)
    return rates


def main():
    R = pl.read_parquet(os.path.join(SCR, "rounds_2026.parquet")).with_columns(
        pl.col("ct_win").cast(pl.Float64))
    tr = R.filter(pl.col("split_primary") == "train")
    te = R.filter(pl.col("split_primary") == "test")
    y_te = te["ct_win"].to_numpy()
    g_rate = float(tr["ct_win"].mean())
    print(f"rounds: train {tr.height}, test {te.height}; global CT rate (train) {g_rate:.4f}")
    p_glob = np.full(len(y_te), g_rate)
    base_round = ll(y_te, p_glob)
    print(f"test log loss, global constant: {base_round:.5f}")

    rates_map = fit_rate(tr, "map_name", 0.0, g_rate)
    p_map = np.array([rates_map.get(m, g_rate) for m in te["map_name"].to_list()])
    print(f"test log loss, per-map constant (raw train rates): {ll(y_te, p_map):.5f}; "
          f"gain {base_round - ll(y_te, p_map):+.5f} nats per round")

    rates_map_s = fit_rate(tr, "map_name", 200.0, g_rate)
    p_map_s = np.array([rates_map_s.get(m, g_rate) for m in te["map_name"].to_list()])
    print(f"test log loss, per-map constant shrunk (prior 200 rounds): {ll(y_te, p_map_s):.5f}; "
          f"gain {base_round - ll(y_te, p_map_s):+.5f}")

    # in-sample gain (upper bound, no held-out)
    rates_all = fit_rate(R, "map_name", 0.0, g_rate)
    p_all = np.array([rates_all.get(m, g_rate) for m in R["map_name"].to_list()])
    yall = R["ct_win"].to_numpy()
    print(f"in-sample per-map constant gain over global (all 28,683 rounds): "
          f"{ll(yall, np.full(len(yall), R['ct_win'].mean())) - ll(yall, p_all):+.5f} nats per round")

    # batch-split per map (CT rate), to see whether a map effect is batch-confounded
    print("\nCT win rate by map and batch (1008 = 2026-10-08 match date), all kept rounds:")
    bt = (R.group_by(["map_name", "batch_1008"]).agg(pl.len().alias("n"), pl.col("ct_win").mean().alias("ct"))
          .sort(["map_name", "batch_1008"]))
    print(bt)

    # post-plant: site constant (cluster) vs map constant, on planted rounds
    Rp = R.filter(pl.col("plant_tick").is_not_null() & pl.col("plant_x").is_not_null())
    Cl = pl.read_parquet(os.path.join(SCR, "planter_clusters_2026_descriptive.parquet")).with_columns(
        pl.col("round").cast(pl.Int64))
    Rp = Rp.join(Cl.select(["match_key", "round", "cluster"]), on=["match_key", "round"], how="inner")
    trp = Rp.filter(pl.col("split_primary") == "train")
    tep = Rp.filter(pl.col("split_primary") == "test")
    yp = tep["ct_win"].to_numpy()
    gp = float(trp["ct_win"].mean())
    rate_mp = fit_rate(trp, "map_name", 0.0, gp)
    pmp = np.array([rate_mp.get(m, gp) for m in tep["map_name"].to_list()])
    rate_ms = fit_rate(trp.with_columns((pl.col("map_name") + "|" + pl.col("cluster").cast(pl.Utf8)).alias("ms")),
                       "ms", 0.0, gp)
    pms = np.array([rate_ms.get(m + "|" + str(c), gp) for m, c in zip(tep["map_name"].to_list(),
                                                                       tep["cluster"].to_list())])
    base_p = ll(yp, np.full(len(yp), gp))
    print(f"\npost-plant rounds: train {trp.height}, test {tep.height}; global CT rate {gp:.4f}")
    print(f"post-plant test log loss: constant {base_p:.5f}; per-map {ll(yp, pmp):.5f} "
          f"(gain {base_p - ll(yp, pmp):+.5f}); per-map x cluster {ll(yp, pms):.5f} "
          f"(gain {base_p - ll(yp, pms):+.5f})")
    print("post-plant CT rate by map x cluster (train):")
    print(trp.group_by(["map_name", "cluster"]).agg(pl.len().alias("n"), pl.col("ct_win").mean().alias("ct"))
          .sort(["map_name", "cluster"]))
    print(f"\nreference: bake-off test log loss {BASE_LL}; gains above are at round level, not state level.")


if __name__ == "__main__":
    main()
