"""D4 extra check: time between a bomb outcome event and the round_end tick of its round.

A defuse or an explosion normally ends the round within a few seconds, so for a correctly attributed outcome
event the gap (round_end.tick - event.tick) should be small and non-negative. The gap is compared between
mismatched and matched outcome rounds (label-based, as in e2_side_check_r13.py), for round 13 and other rounds.
ASSUMPTION A9: 64 ticks = 1 s (tick rate verified by second * 64 == tick in bomb_state).
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\D4_gap_check.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(14)

ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
OUTCOMES = ["bomb_defused", "bomb_exploded"]
T, CT = 2, 3


def read(ch, cols):
    fd = sorted(BASE.glob(f"basic_{ch}.*"))[0]
    fs = sorted(glob.glob(str(fd / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


re_ = (read("round_end", ["match_key", "round", "tick", "winner_team_code", "player_count"])
       .unique(["match_key", "round"], keep="first").filter(pl.col("player_count") == 20)
       .rename({"tick": "end"}))
bs = read("bomb_state", ["match_key", "round", "tick", "event_type"]).filter(pl.col("event_type").is_in(OUTCOMES))
ev = bs.join(re_, on=["match_key", "round"], how="inner").with_columns(
    (pl.col("end") - pl.col("tick")).alias("gap_ticks"),
    pl.when(pl.col("event_type") == "bomb_defused").then(pl.lit(CT)).otherwise(pl.lit(T)).alias("bomb_side"),
)
# grouped mismatch per (match, round): same rule as e2_side_check_r13.py
g = (ev.group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs"),
                                              pl.col("winner_team_code").first(), pl.col("end").first(),
                                              pl.col("tick").min().alias("first_tick"),
                                              pl.col("tick").max().alias("last_tick"))
     .with_columns(
         pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(CT)).otherwise(pl.lit(T)).alias("bomb_side"))
     .with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mm"),
                   (pl.col("end") - pl.col("last_tick")).alias("gap_last"),
                   (pl.col("end") - pl.col("first_tick")).alias("gap_first"),
                   pl.when(pl.col("round") == 13).then(pl.lit("r13")).otherwise(pl.lit("other")).alias("rbin")))
print(f"outcome rounds: {g.height}; mismatches: {int(g['mm'].sum())}")
print("\nround_end.tick minus last outcome event tick (ticks), by rbin and mismatch:")
print(g.group_by(["rbin", "mm"]).agg(
    pl.len().alias("n"),
    pl.col("gap_last").median().alias("median_gap"),
    pl.col("gap_last").quantile(0.1).alias("p10"),
    pl.col("gap_last").quantile(0.9).alias("p90"),
    pl.col("gap_last").min().alias("min"),
    pl.col("gap_last").max().alias("max"),
    (pl.col("gap_last") < 0).sum().alias("n_event_after_end"),
    (pl.col("gap_last").abs() > 640).sum().alias("n_gap_over_10s"),
).sort(["rbin", "mm"]))
print("\nmismatched rounds only: event type x winner_team_code (round_end) x round bin")
print(g.filter(pl.col("mm")).group_by(["rbin", "evs", "winner_team_code"]).len().sort(["rbin", "evs", "winner_team_code"]))
