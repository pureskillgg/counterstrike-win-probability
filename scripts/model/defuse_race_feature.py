"""Defuse-race margin for post-plant seconds (side quest; first version, straight-line distance).

The margin is the fuse time left minus the time the nearest living CT needs to reach the bomb and defuse it:

  fuse_left   = 40 - seconds_since_plant                       (C4 fuse, 40 s)
  ct_travel   = min over living CTs of distance(xy, bomb centre) / SPEED   (SPEED = 250 units per second, a crude run speed)
  defuse_time = 10                                              (defuse without a kit; kits are not used in this first version)
  margin      = fuse_left - (ct_travel + defuse_time)           (positive: a defuse is still physically possible)

The paper (Huang et al. 2026) uses navigation-mesh travel times; straight-line distance here is a stand-in. Bomb centre: the plant
position of the same round (data/positions_1hz/_plants). Only de_mirage, only post-plant seconds, 2026 days.

  ...python.exe -I -X utf8 scripts\\model\\defuse_race_feature.py
Writes data/diagnostics/defuse_race_de_mirage.parquet and prints a sanity table (CT win rate by margin sign).
"""
import glob
import json
import os
import sys

import numpy as np
import polars as pl

ROOT = r"G:\pureskill\csds-cs2"
SPEED = 250.0
FUSE = 40.0
DEFUSE = 10.0
CT = 3
OUT = os.path.join(ROOT, "data", "diagnostics", "defuse_race_de_mirage.parquet")


def main():
    grid = pl.read_parquet(os.path.join(ROOT, "data", "combined", "grid.parquet"),
                           columns=["match_key", "round", "second", "bomb_planted", "seconds_since_plant", "map_name", "era", "upload_day"])
    grid = grid.filter((pl.col("era") == "2026") & (pl.col("map_name") == "de_mirage") & (pl.col("bomb_planted") == 1))
    rounds = pl.read_parquet(os.path.join(ROOT, "data", "combined", "rounds.parquet"), columns=["match_key", "round", "final"])
    ct_win = rounds.with_columns((pl.col("final") == 3).cast(pl.Int8).alias("ct_win")).select("match_key", "round", "ct_win")
    days = sorted(grid["upload_day"].unique().to_list())
    out = []
    for day in days:
        pos_f = os.path.join(ROOT, "data", "positions_1hz", day.replace("/", "-") + ".parquet")
        plant_f = os.path.join(ROOT, "data", "positions_1hz", "_plants", day.replace("/", "-") + ".parquet")
        if not (os.path.exists(pos_f) and os.path.exists(plant_f)):
            continue
        g = grid.filter(pl.col("upload_day") == day).select("match_key", "round", "second", "seconds_since_plant")
        if g.height == 0:
            continue
        plants = (pl.read_parquet(plant_f).filter(pl.col("x").is_not_null() & pl.col("y").is_not_null())
                  .group_by(["match_key", "round"]).agg(pl.col("x").first().alias("bx"), pl.col("y").first().alias("by")))
        pos = pl.read_parquet(pos_f, columns=["match_key", "round", "second", "side", "x", "y"]).filter(pl.col("side") == CT)
        j = (pos.join(g, on=["match_key", "round", "second"], how="inner")
             .join(plants, on=["match_key", "round"], how="inner"))
        if j.height == 0:
            continue
        j = j.with_columns(((pl.col("x") - pl.col("bx")) ** 2 + (pl.col("y") - pl.col("by")) ** 2).sqrt().alias("dist"))
        per = (j.group_by(["match_key", "round", "second", "seconds_since_plant"])
               .agg(pl.col("dist").min().alias("nearest_ct_dist")))
        out.append(per)
        print(day, per.height, "post-plant seconds", flush=True)
    if not out:
        raise SystemExit("no post-plant seconds found")
    D = pl.concat(out).with_columns(
        (FUSE - pl.col("seconds_since_plant")).alias("fuse_left"),
        (pl.col("nearest_ct_dist") / SPEED).alias("ct_travel"),
    ).with_columns((pl.col("fuse_left") - (pl.col("ct_travel") + DEFUSE)).alias("defuse_margin"))
    D = D.join(ct_win, on=["match_key", "round"], how="inner")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    D.write_parquet(OUT, compression="zstd")
    feasible = D.filter(pl.col("defuse_margin") > 0)
    infeasible = D.filter(pl.col("defuse_margin") <= 0)
    summary = {
        "rows": D.height, "rounds": D.select("match_key", "round").unique().height,
        "ct_win_rate_margin_positive": float(feasible["ct_win"].mean()) if feasible.height else None,
        "rows_margin_positive": feasible.height,
        "ct_win_rate_margin_nonpositive": float(infeasible["ct_win"].mean()) if infeasible.height else None,
        "rows_margin_nonpositive": infeasible.height,
        "note": "first version: straight-line distance, 250 units/s, 40 s fuse, 10 s defuse; no defuse-kit distinction",
    }
    print(json.dumps(summary, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
