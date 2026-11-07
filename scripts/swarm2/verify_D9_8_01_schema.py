"""verify D9-8 step 1: schema and value probe for the 2024 tome channels used by the round-13 claim.

Read-only. Reads parquet metadata and projected columns only.
Output goes to stdout; the caller tees it into logs\\swarm2\\verify_D9_8.log.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(120)

ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"


def files_of(channel):
    fd = BASE / f"basic_{channel}{SUF}"
    return sorted(glob.glob(str(fd / "dataframe_*")))


for ch in ["round_end", "bomb_state", "round_start", "round_state"]:
    fs = files_of(ch)
    print(f"\n=== {ch}: {len(fs)} dataframe files")
    if not fs:
        continue
    schema = pl.read_parquet_schema(fs[0])
    print("schema:", dict(schema))

ro = pl.scan_parquet(files_of("round_end")).select(
    ["match_key", "round", "winner_team_code", "win_reason_code", "win_reason_message",
     "player_count", "tick", "second", "legacy_code"]
)
print("\nround_end rows (lazy count):", ro.select(pl.len()).collect().item())
print("round_end sample:")
print(ro.head(5).collect())
print("\nround_end win_reason_message x winner_team_code x win_reason_code counts (all rows):")
print(
    ro.group_by(["win_reason_message", "winner_team_code", "win_reason_code"])
    .agg(pl.len().alias("n"))
    .sort(["win_reason_message", "winner_team_code", "win_reason_code"])
    .collect()
)
print("\nplayer_count values:")
print(ro.group_by("player_count").agg(pl.len().alias("n")).sort("player_count").collect())

bs = pl.scan_parquet(files_of("bomb_state"))
print("\nbomb_state columns:", bs.collect_schema().names())
print("bomb_state event_type counts:")
print(bs.group_by("event_type").agg(pl.len().alias("n")).sort("event_type").collect())

rs = pl.scan_parquet(files_of("round_start"))
print("\nround_start columns:", rs.collect_schema().names())
print(rs.head(3).collect())

ms = pl.scan_parquet(files_of("round_end")).select(pl.col("match_key").str.slice(0, 120)).head(3).collect()
print("\nmatch_key sample:")
print(ms)
