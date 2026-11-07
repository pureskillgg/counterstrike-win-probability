"""Verify D7-3 (independent): coverage of round_end / bomb events by match day, key-path day vs header UTC day.
Reads only round_end and header (projected columns). ASSUMPTION: 20-player rounds (player_count == 20) are full-match rounds.
"""
import glob, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(240)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
def files(fd): return sorted(glob.glob(str(fd / "dataframe_*")))

H = pl.read_parquet(str(BASE / "header" / "dataframe_00000"), columns=["key", "match_date", "build_num"]).rename({"key": "match_key"})
H = H.with_columns(
    pl.col("match_date").str.replace("Z$", "").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f", strict=False).alias("mdt"),
    pl.col("match_key").str.extract(r"csds/(\d{4}/\d\d/\d\d)/", 1).alias("key_day"),
)
H = H.with_columns(pl.col("mdt").dt.strftime("%Y/%m/%d").alias("utc_day"))
print("header keys with key_day null:", H.filter(pl.col("key_day").is_null()).height)
print("header matches where utc_day != key_day:", H.filter(pl.col("utc_day") != pl.col("key_day")).height, "of", H.height)
print("header key_day counts:", H.group_by("key_day").len().sort("key_day").rows())

RE = pl.concat([pl.scan_parquet(f).select(["match_key", "round", "winner_team_code", "player_count"]) for f in files(BASE / f"basic_round_end{SUF}")]).collect()
print("\nround_end rows:", RE.height, "| distinct matches:", RE["match_key"].n_unique())
print("round_end matches not in header:", RE.filter(~pl.col("match_key").is_in(H["match_key"].implode()))["match_key"].n_unique())
mk = RE.select("match_key").unique().join(H.select(["match_key", "key_day", "utc_day", "build_num"]), on="match_key", how="left")
print("matches with round_end rows, by key_day (count):", mk.group_by("key_day").len().sort("key_day").rows())
print("header matches with NO round_end rows:", H.join(RE.select("match_key").unique(), on="match_key", how="anti").height)
miss = H.join(RE.select("match_key").unique(), on="match_key", how="anti")
print("  of which build counts:", miss.group_by("build_num").len().rows(), "| utc_day counts:", miss.group_by("utc_day").len().sort("utc_day").rows())

BS = pl.concat([pl.scan_parquet(f).select(["match_key", "round", "event_type"]) for f in files(BASE / f"basic_bomb_state{SUF}")]).collect()
print("\nbomb_state event_types:", BS.group_by("event_type").len().sort("event_type").rows())
OUT = BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"])).select(["match_key", "round"]).unique()
print("matches with a bomb outcome row:", OUT["match_key"].n_unique())
cov = H.join(OUT.select("match_key").unique(), on="match_key", how="left").with_columns(pl.col("match_key").is_not_null().alias("has_outcome"))
print("header matches with >=1 bomb outcome, by key_day:", cov.group_by(["key_day", "has_outcome"]).len().sort(["key_day", "has_outcome"]).rows())
print("\nheader matches by key_day x build:", H.group_by(["key_day", "build_num"]).len().sort(["key_day", "build_num"]).rows())
print("\nround_end rows (player_count==20) round==13 by key_day:", RE.filter((pl.col("player_count") == 20) & (pl.col("round") == 13)).join(H.select(["match_key", "key_day"]), on="match_key", how="left").group_by("key_day").len().sort("key_day").rows())
print("\nround_end row count by build (player_count==20):", RE.filter(pl.col("player_count") == 20).join(H.select(["match_key", "build_num"]), on="match_key", how="left").group_by("build_num").len().rows())
