"""A11 (read-only): header freeze_time distribution and round-1 freeze-end tick for tick-66 vs other matches by batch."""
import glob
from pathlib import Path
import polars as pl

TOME = Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds")
SUFFIX = ".2026-09-18,2026-10-08"

def read(ch, cols):
    folder = TOME / ("header" if ch == "header" else f"basic_{ch}{SUFFIX}")
    return pl.scan_parquet(sorted(glob.glob(str(folder / "dataframe_*")))).select(cols).collect()

h = read("header", ["key", "match_date", "freeze_time", "round_time_defuse"]).with_columns(day=pl.col("match_date").str.slice(0, 10))
print("header freeze_time (value, matches):", h.group_by("freeze_time").len().sort("freeze_time").rows())
rst = read("round_start", ["match_key", "round", "tick"])
rs = read("round_state", ["match_key", "round", "tick", "event_type"])
ls = (rs.filter(pl.col("event_type") == "round_freeze_end").group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("fe")))
j = rst.join(ls, on=["match_key", "round"], how="left").join(h.select(["key", "day", "freeze_time"]), left_on="match_key", right_on="key", how="left")
j = j.with_columns(gap_s=(pl.col("fe") - pl.col("tick")) / 64, is66=pl.col("tick") == 66,
                   batch=pl.when(pl.col("day") == "2026-10-08").then(pl.lit("10-08")).otherwise(pl.lit("other")))
print("gap_s (freeze_end - round_start, seconds) by freeze_time, all rounds: median")
print(j.group_by("freeze_time").agg(pl.col("gap_s").median().alias("median_gap_s"), pl.len().alias("n")).sort("freeze_time").rows())
print("round 1 only, by batch and tick66: (batch, tick66, rows, median start tick, median fe tick, median gap s)")
r1 = j.filter(pl.col("round") == 1)
print(r1.group_by(["batch", "is66"]).agg(pl.len().alias("n"), pl.col("tick").median().alias("start_med"),
      pl.col("fe").median().alias("fe_med"), pl.col("gap_s").median().alias("gap_med")).sort(["batch", "is66"]).rows())
print("rounds 2+ only, by batch: (batch, rows, median start tick, median fe tick, median gap s)")
r2 = j.filter(pl.col("round") >= 2)
print(r2.group_by("batch").agg(pl.len().alias("n"), pl.col("tick").median().alias("start_med"),
      pl.col("fe").median().alias("fe_med"), pl.col("gap_s").median().alias("gap_med")).sort("batch").rows())
