"""Random ticks per round, for the random-tick sweep. For each round, K = 64 ticks are drawn uniformly at random between freeze end and the
round's effective end (the same live range as the grid, including the 116 s cap for unplanted rounds). Draws are kept in draw order, so
the first n of the 64 are a random subset of size n, and the subsets are nested across n. Each drawn tick gets the basic model's 13 event
inputs, computed exactly as build_grid_step.py computes them at a grid tick (alive counts from the deaths table, bomb state from the plant
tick, time since live start and since plant, map one-hots).

  ...python.exe -I -X utf8 scripts\\data\\build_random_tick_pool.py --seed 0 [--store combined]
Writes data/<store>/random_tick_pool_s<seed>.parquet (columns: match_key, round, tick, draw_rank, the 13 inputs, label_ct_win, era).
"""
import argparse
import os
import sys
import time

import numpy as np
import polars as pl
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"G:\pureskill\csds-cs2"
T, CT = 2, 3
TPS = 64
BIG = 1 << 31
K = 64
CANDIDATES = 80
CHUNK = 20000
MAPS = ["ancient", "anubis", "cache", "dust2", "inferno", "mirage", "nuke"]


def keep_first_distinct(cand, k):
    """cand: (n, C) draws in draw order. Keep the first k distinct values in each row, in draw order."""
    n, C = cand.shape
    eq = cand[:, :, None] == cand[:, None, :]
    earlier = np.tril(np.ones((C, C), dtype=bool), k=-1)  # earlier[i, j] is True for j < i
    dup = (eq & earlier[None, :, :]).any(axis=2)
    first = ~dup
    return first & (np.cumsum(first, axis=1) <= k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--store", default="combined")
    ap.add_argument("--grid", action="store_true", help="control: draw from the grid's one-second ticks (freeze + 64k) instead of any tick")
    a = ap.parse_args()
    d = os.path.join(ROOT, "data", a.store)
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    prefix = "random_second_pool" if a.grid else "random_tick_pool"

    R = pl.read_parquet(os.path.join(d, "rounds.parquet")).filter(pl.col("use")).sort(["match_key", "round"])
    M = pl.read_parquet(os.path.join(d, "matches.parquet")).select("match_key", "map_name")
    R = R.join(M, on="match_key", how="left").with_columns(pl.int_range(pl.len()).alias("ridx"))
    DE = (pl.read_parquet(os.path.join(d, "deaths.parquet"))
          .join(R.select("match_key", "round", "ridx"), on=["match_key", "round"], how="inner").filter(pl.col("side_ok")))

    freeze = R["freeze"].to_numpy().astype(np.int64)
    end = R["end_eff"].to_numpy().astype(np.int64)
    plant = R["plant_tick"].fill_null(-1).to_numpy().astype(np.int64)
    end = np.where(plant >= 0, end, np.minimum(end, freeze + 116 * TPS))
    length = end - freeze
    mk = R["match_key"].to_numpy()
    rnd = R["round"].to_numpy().astype(np.int64)
    label = (R["final"].to_numpy().astype(np.int64) == CT).astype(np.int8)
    era = R["era"].to_numpy()
    mapn = R["map_name"].to_numpy()

    keys = {}
    for side in (CT, T):
        s = DE.filter(pl.col("side") == side)
        keys[side] = np.sort(s["ridx"].to_numpy().astype(np.int64) * BIG + s["tick"].to_numpy().astype(np.int64))

    def deaths_before(side, ridx, ticks):
        hi = np.searchsorted(keys[side], ridx * BIG + ticks, side="right")
        lo = np.searchsorted(keys[side], ridx * BIG, side="left")
        return hi - lo

    nR = R.height
    out_path = os.path.join(d, f"{prefix}_s{a.seed}.parquet")
    tmp = out_path + ".tmp"
    writer, rows = None, 0
    for c0 in range(0, nR, CHUNK):
        c1 = min(nR, c0 + CHUNK)
        f, L = freeze[c0:c1], length[c0:c1]
        ok = L > 0
        if a.grid:  # control: the grid's seconds, freeze + 64k with k < ceil(L / 64)
            nsec = np.maximum(1, -(-np.maximum(L, 1) // TPS))
            cand = f[:, None] + TPS * np.floor(rng.random((c1 - c0, CANDIDATES)) * nsec[:, None]).astype(np.int64)
        else:
            cand = f[:, None] + np.floor(rng.random((c1 - c0, CANDIDATES)) * np.maximum(L, 1)[:, None]).astype(np.int64)
        keep = keep_first_distinct(cand, K) & ok[:, None]
        rr, cc = np.nonzero(keep)
        if rr.size == 0:
            continue
        rank = (np.cumsum(keep, axis=1) - 1)[rr, cc]
        ridx = rr + c0
        ticks = cand[rr, cc]
        ac = 5 - deaths_before(CT, ridx, ticks)
        at = 5 - deaths_before(T, ridx, ticks)
        pl_at = plant[ridx]
        planted = (pl_at >= 0) & (ticks >= pl_at)
        G = pl.DataFrame({
            "match_key": mk[ridx], "round": rnd[ridx], "tick": ticks, "draw_rank": rank.astype(np.int16),
            "alive_ct": ac.astype(np.int8), "alive_t": at.astype(np.int8), "ct_alive_minus_t": (ac - at).astype(np.int8),
            "bomb_planted": planted.astype(np.int8),
            "seconds_since_round_live": ((ticks - freeze[ridx]) / TPS).astype(np.float32),
            "seconds_since_plant": np.where(planted, (ticks - pl_at) / TPS, -1.0).astype(np.float32),
            "label_ct_win": label[ridx], "era": era[ridx], "map_name": mapn[ridx],
        })
        for m in MAPS:
            G = G.with_columns((pl.col("map_name") == f"de_{m}").cast(pl.Float32).alias(f"map_is_{m}"))
        G = G.drop("map_name")
        tab = G.to_arrow()
        if writer is None:
            writer = pq.ParquetWriter(tmp, tab.schema, compression="zstd")
        writer.write_table(tab.cast(writer.schema))
        rows += G.height
        print(f"  rounds {c1}/{nR}  rows {rows:,}  ({time.time() - t0:.0f}s)", flush=True)
    writer.close()
    os.replace(tmp, out_path)
    print(f"done: seed {a.seed}, {rows:,} random-tick rows, {nR} rounds ({time.time() - t0:.0f}s) -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
