"""Feasibility probe (Task A follow-up): are the 1 Hz positions continuous per player?

Read-only on data\\state_sample\\players_1hz.parquet. For consecutive 1 Hz snapshots of the same
player (same match, round, player_id), the XY displacement per second should stay near running speed.
Reports displacement quantiles, the share above a few speed caps, and the k=1 displacement by team.
Run from the repo root:
  pixi run python -I -X utf8 scripts\\eda\\feas_probe_speed.py
"""

import os
import sys

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")

ROOT = r"G:\pureskill\csds-cs2"
SAMPLE = os.path.join(ROOT, "data", "state_sample")
KEYS = ["match_key", "round"]


def main():
    P = (pl.read_parquet(os.path.join(SAMPLE, "players_1hz.parquet"),
                         columns=["match_key", "round", "tick", "seconds_since_live", "player_id",
                                  "team_code", "x_pos", "y_pos"])
         .with_columns(pl.col("round").cast(pl.Int64), pl.col("x_pos").cast(pl.Float64),
                       pl.col("y_pos").cast(pl.Float64)))
    print(f"rows {P.height}; null x/y: {P['x_pos'].null_count()} / {P['y_pos'].null_count()}")
    P = P.sort(KEYS + ["player_id", "seconds_since_live"])
    nxt = P.with_columns(
        pl.col("seconds_since_live").shift(-1).over(KEYS + ["player_id"]).alias("k_next"),
        pl.col("x_pos").shift(-1).over(KEYS + ["player_id"]).alias("x_next"),
        pl.col("y_pos").shift(-1).over(KEYS + ["player_id"]).alias("y_next"))
    pairs = nxt.filter(pl.col("k_next") == pl.col("seconds_since_live") + 1)
    d = ((pairs["x_next"] - pairs["x_pos"]) ** 2 + (pairs["y_next"] - pairs["y_pos"]) ** 2).sqrt().to_numpy()
    print(f"consecutive 1 s pairs for the same player: {len(d)}")
    print(f"XY displacement per second: median {np.median(d):.1f}, p90 {np.percentile(d, 90):.1f}, "
          f"p99 {np.percentile(d, 99):.1f}, max {d.max():.1f}")
    for cap in (260, 300, 400, 600):
        print(f"  share of 1 s steps > {cap} units: {float(np.mean(d > cap)):.5f} ({int((d > cap).sum())})")

    fz = P.filter(pl.col("seconds_since_live") == 0).select(KEYS + ["player_id", "team_code", "x_pos", "y_pos"]) \
        .rename({"x_pos": "x0", "y_pos": "y0", "team_code": "tc0"})
    k1 = (P.filter(pl.col("seconds_since_live") == 1).join(fz, on=KEYS + ["player_id"], how="inner")
          .with_columns(((pl.col("x_pos") - pl.col("x0")) ** 2 + (pl.col("y_pos") - pl.col("y0")) ** 2)
                        .sqrt().alias("dist")))
    print("\nk=1 displacement from freeze, by team (2 = T, 3 = CT):")
    for tc in (2, 3):
        sub = k1.filter(pl.col("team_code") == tc)["dist"].to_numpy()
        print(f"  team {tc}: pairs {len(sub)}; share > 50: {float(np.mean(sub > 50)):.4f}; "
              f"median {np.median(sub):.1f}; p10 {np.percentile(sub, 10):.1f}; p90 {np.percentile(sub, 90):.1f}")
    zero = k1["dist"].to_numpy()
    print(f"  exact zero displacement at k=1 (all teams): {int((zero == 0).sum())} of {len(zero)}")


if __name__ == "__main__":
    main()
