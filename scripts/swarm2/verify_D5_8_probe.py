"""Verifier probe for D5-8: schema and event-type census for the 2024 tome (lazy, projected)."""
import glob
import os
import sys

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(80)

BASE = r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"
SUF = ".2023-12-28,2024-02-25"


def scan(ch):
    fd = os.path.join(BASE, f"basic_{ch}{SUF}")
    return pl.scan_parquet(sorted(glob.glob(os.path.join(fd, "dataframe_*"))), missing_columns="insert",
                           extra_columns="ignore")


rs = scan("round_state")
print("round_state rows:", rs.select(pl.len()).collect().item())
print(rs.group_by("event_type").agg(pl.len().alias("n")).sort("n", descending=True).collect())
print(rs.group_by(["event_type", "phase"]).agg(pl.len().alias("n")).sort("n", descending=True).head(40).collect())
bs = scan("bomb_state")
print("bomb_state event types:")
print(bs.group_by("event_type").agg(pl.len().alias("n")).collect())
re_ = scan("round_end")
print("round_end rows:", re_.select(pl.len()).collect().item())
print(re_.group_by(["win_reason_code", "win_reason_message"]).agg(pl.len().alias("n")).sort("n", descending=True).collect())
print(re_.group_by("player_count").agg(pl.len().alias("n")).sort("n", descending=True).collect())
print("round_state is_warmup:")
print(rs.group_by("is_warmup").agg(pl.len().alias("n")).collect())
