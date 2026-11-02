"""State grid at any tick step, for the grid-resolution comparison. Same rules as make_grid.py (effective end, 116 s cap for unplanted
rounds, deaths from the deaths table, round weights 1/n), built in chunks of rounds so only one chunk is in memory at a time.

  ...python.exe -I -X utf8 scripts\\data\\build_grid_step.py --step 16 [--store combined]

--step 64 reproduces the one-second grid (rows equal make_grid.py's). --step 16 is a quarter second. Seconds columns always use 64 ticks
per second. Writes data/<store>/grid_step<N>.parquet with the columns of the basic model (no loadouts needed).
"""
import argparse
import os
import sys
import time

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import numpy as np
import polars as pl
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"G:\pureskill\csds-cs2"
T, CT = 2, 3
TPS = 64  # ticks per second
BIG = 1 << 31


def build_chunk(R, DE, step):
    """R: rounds of one chunk (ridx 0..n-1 local), DE: its deaths with local ridx. Returns the rows of the chunk (polars)."""
    nR = R.height
    freeze = R["freeze"].to_numpy().astype(np.int64)
    end = R["end_eff"].to_numpy().astype(np.int64)
    plant0 = R["plant_tick"].fill_null(-1).to_numpy().astype(np.int64)
    end = np.where(plant0 >= 0, end, np.minimum(end, freeze + 116 * TPS))
    n = np.maximum(0, -(-(end - freeze) // step)).astype(np.int64)
    total = int(n.sum())
    if total == 0:
        return None
    ridx = np.repeat(np.arange(nR, dtype=np.int64), n)
    start = np.cumsum(n) - n
    k = np.arange(total, dtype=np.int64) - np.repeat(start, n)
    ts = freeze[ridx] + step * k

    def count_deaths(side):
        s = DE.filter(pl.col("side") == side)
        keys = np.sort(s["ridx"].to_numpy().astype(np.int64) * BIG + s["tick"].to_numpy().astype(np.int64))
        hi = np.searchsorted(keys, ridx * BIG + ts, side="right")
        lo = np.searchsorted(keys, ridx * BIG, side="left")
        return hi - lo

    dct, dt = count_deaths(CT), count_deaths(T)
    ac, at = 5 - dct, 5 - dt
    plant = plant0[ridx]
    planted = (plant >= 0) & (ts >= plant)
    label = (R["final"].to_numpy().astype(np.int64) == CT).astype(np.int8)[ridx]
    return pl.DataFrame({
        "match_key": R["mk_id"].to_numpy().astype(np.int32)[ridx],  # integer match id (see the matches table next to the grid)
        "round": R["round"].to_numpy().astype(np.int64)[ridx],
        "tick": ts,
        "second": k.astype(np.int32),
        "alive_ct": ac.astype(np.int8), "alive_t": at.astype(np.int8), "ct_alive_minus_t": (ac - at).astype(np.int8),
        "bomb_planted": planted.astype(np.int8),
        "seconds_since_round_live": ((ts - freeze[ridx]) / TPS).astype(np.float32),
        "seconds_since_plant": np.where(planted, (ts - plant) / TPS, -1.0).astype(np.float32),
        "label_ct_win": label,
        "w_round": (1.0 / np.maximum(n, 1))[ridx].astype(np.float32),
        "era": pl.Series(R["era"].to_numpy()[ridx]).cast(pl.Categorical),
        "map_name": pl.Series(R["map_name"].to_numpy()[ridx]).cast(pl.Categorical),
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=16, help="ticks between rows: 64 = one second, 16 = quarter second")
    ap.add_argument("--store", default="combined")
    ap.add_argument("--chunk", type=int, default=20000, help="rounds per chunk")
    ap.add_argument("--subset", default=None, help="parquet with an mk_id column: build only these matches (mk_id numbering is unchanged)")
    ap.add_argument("--suffix", default="", help="output name suffix: grid_step<N><suffix>.parquet")
    a = ap.parse_args()
    d = os.path.join(ROOT, "data", a.store)
    t0 = time.time()
    R = pl.read_parquet(os.path.join(d, "rounds.parquet")).filter(pl.col("use")).sort(["match_key", "round"])
    M = pl.read_parquet(os.path.join(d, "matches.parquet")).select("match_key", "map_name")
    R = R.join(M, on="match_key", how="left")
    ids = pl.DataFrame({"match_key": R["match_key"].unique().sort()}).with_columns(pl.int_range(pl.len(), dtype=pl.Int32).alias("mk_id"))
    matches = ids.join(R.group_by("match_key").agg(pl.col("era").first()), on="match_key", how="left").with_columns(pl.lit("").alias("match_date"))
    R = R.join(ids, on="match_key", how="left")
    if a.subset:
        keep = pl.read_parquet(a.subset, columns=["mk_id"])["mk_id"].to_list()
        R = R.filter(pl.col("mk_id").is_in(keep))
        print(f"subset: {len(keep)} matches, {R.height:,} rounds", flush=True)
    DE = pl.read_parquet(os.path.join(d, "deaths.parquet"))
    R = R.with_columns(pl.int_range(pl.len()).alias("ridx"))
    DE = DE.join(R.select("match_key", "round", "ridx"), on=["match_key", "round"], how="inner").filter(pl.col("side_ok"))
    nR = R.height
    matches.write_parquet(os.path.join(d, f"grid_step{a.step}_matches.parquet"))
    out = os.path.join(d, f"grid_step{a.step}{a.suffix}.parquet")
    tmp = out + ".tmp"
    writer, rows, matches = None, 0, set()
    for c0 in range(0, nR, a.chunk):
        c1 = min(nR, c0 + a.chunk)
        Rc = R.slice(c0, c1 - c0).with_columns((pl.col("ridx") - c0).alias("ridx"))
        DEc = DE.filter((pl.col("ridx") >= c0) & (pl.col("ridx") < c1)).with_columns((pl.col("ridx") - c0).alias("ridx"))
        G = build_chunk(Rc, DEc, a.step)
        if G is None:
            continue
        rows += G.height
        matches.update(Rc["match_key"].unique().to_list())
        tab = G.to_arrow()
        if writer is None:
            writer = pq.ParquetWriter(tmp, tab.schema, compression="zstd")
        writer.write_table(tab.cast(writer.schema))
        print(f"  rounds {c1}/{nR}  rows {rows:,}  ({time.time() - t0:.0f}s)", flush=True)
    writer.close()
    os.replace(tmp, out)
    print(f"done: step {a.step} ticks, {rows:,} rows, {len(matches)} matches, {nR} rounds ({time.time() - t0:.0f}s) -> {out}", flush=True)


if __name__ == "__main__":
    main()
