"""Where do bomb_begin_plant events sit relative to the completed plant (grid bomb_planted onset)?

Read-only, laptop. Question: after the completed plant, are there begin events, and do rounds
with several begin events put the bomb at a different site than the completed plant?
"""
import glob
import os
import sys

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"G:\pureskill\csds-cs2"
grid = os.path.join(ROOT, "data", "combined", "grid.parquet")
plants = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(os.path.join(ROOT, "data", "positions_1hz", "_plants", "*.parquet")))],
                   how="vertical_relaxed").with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64))
print("begin events", plants.height, "rounds with begin", plants.select("match_key", "round").unique().height, flush=True)

done = (pl.scan_parquet(grid).select("match_key", "round", "tick", "bomb_planted", "era")
        .filter((pl.col("era") == "2026") & (pl.col("bomb_planted") == 1))
        .group_by("match_key", "round").agg(pl.col("tick").min().alias("p_grid"))
        .with_columns(pl.col("round").cast(pl.Int64), pl.col("p_grid").cast(pl.Int64))
        .collect())
print("rounds with a completed plant in grid (2026)", done.height, flush=True)

b = plants.join(done, on=["match_key", "round"], how="inner")
rounds_b = b.select("match_key", "round", "p_grid").unique()
print("rounds with begins AND completed plant", rounds_b.height, flush=True)

# begin events by timing relative to the completed plant (grid resolution is 1 s, so allow a 64-tick margin)
b = b.with_columns(
    (pl.col("tick") > pl.col("p_grid") + 64).alias("after"),
    (pl.col("tick") < pl.col("p_grid") - 64).alias("before"))
tot = b.height
n_after = int(b["after"].sum())
print(f"begin events: {tot}; after completed plant (+1 s): {n_after}; before: {int(b['before'].sum())}; within 1 s: {tot - n_after - int(b['before'].sum())}", flush=True)
rounds_after = b.filter(pl.col("after")).select("match_key", "round").unique().height
print("rounds with a begin event after the completed plant:", rounds_after, flush=True)

# multi-begin rounds, and whether the LAST begin before completion sits at a different site than the completed plant's begin
g = b.sort("tick").group_by("match_key", "round", maintain_order=True).agg(
    pl.len().alias("n_begin"), pl.col("p_grid").first(),
    pl.col("x").last().alias("x_last"), pl.col("y").last().alias("y_last"),
    pl.col("x").first().alias("x_first"), pl.col("y").first().alias("y_first"),
    pl.col("tick").first().alias("t_first"), pl.col("tick").last().alias("t_last"))
multi = g.filter(pl.col("n_begin") > 1)
print("rounds with >1 begin event", multi.height, flush=True)
if multi.height:
    d = ((multi["x_last"] - multi["x_first"]) ** 2 + (multi["y_last"] - multi["y_first"]) ** 2).sqrt()
    print("first->last begin displacement (units): median", round(float(d.median()), 1), "p90", round(float(d.quantile(0.9)), 1),
          "| share >100 u", round(float((d > 100).mean()), 3), flush=True)
    print("median seconds from first begin to completed plant", round(float(((multi["p_grid"] - multi["t_first"]) / 64).median()), 2), flush=True)
    print("median seconds from last begin to completed plant", round(float(((multi["p_grid"] - multi["t_last"]) / 64).median()), 2), flush=True)
    print("share of multi-begin rounds where last begin is after completion", round(float((multi["t_last"] > multi["p_grid"] + 64).mean()), 4), flush=True)
