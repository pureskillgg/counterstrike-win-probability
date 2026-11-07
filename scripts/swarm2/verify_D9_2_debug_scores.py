"""verify_D9_2 debug: why the score test shows only build 9961; calendar-date overlap of builds.

Reads the scratch parquet written by verify_D9_2_core.py (no tome access needed except header).
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(50)
SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")

R = pl.read_parquet(SCR / "verify_D9_2_round.parquet")
H = pl.scan_parquet(sorted(glob.glob(str(BASE / "header" / "dataframe_*")))).select(
    ["key", "match_date", "build_num", "t_starters_score_final", "ct_starters_score_final"]).collect()
H = H.with_columns(pl.col("key").alias("match_key"), pl.col("match_date").str.slice(0, 10).alias("date_utc"))

print("== round_end rows by build (R), round min/max per match")
pm = (R.group_by("match_key").agg(pl.len().alias("n_rounds"), pl.col("round").min().alias("min_round"),
                                  pl.col("round").max().alias("max_round"), pl.col("build_num").first())
        .join(H.select(["match_key", "t_starters_score_final", "ct_starters_score_final", "date_utc"]),
              on="match_key", how="left"))
print("matches per build:", pm.group_by("build_num").len().sort("build_num").rows())
print("min_round distribution by build:")
print(pm.group_by(["build_num", "min_round"]).len().sort(["build_num", "min_round"]).rows())
print("max_round (top) by build:")
print(pm.group_by(["build_num", "max_round"]).len().sort(["build_num", "max_round"]).tail(30).rows())
pm = pm.with_columns((pl.col("t_starters_score_final") + pl.col("ct_starters_score_final")).alias("hdr_sum"))
pm = pm.with_columns((pl.col("hdr_sum") - pl.col("n_rounds")).alias("hdr_minus_rounds"))
print("header score sum minus number of round_end rows, by build (value: count):")
print(pm.group_by(["build_num", "hdr_minus_rounds"]).len().sort(["build_num", "hdr_minus_rounds"]).rows())
print("matches where max_round > 24 by build:",
      pm.filter(pl.col("max_round") > 24).group_by("build_num").len().rows())
print("matches with min_round != 1 by build:",
      pm.filter(pl.col("min_round") != 1).group_by("build_num").len().rows())

print("\n== calendar (match_date UTC) by build: min and max date, and counts by month")
hb = H.select(["match_key", "build_num", "date_utc", "match_date"])
print(hb.group_by("build_num").agg(pl.col("date_utc").min().alias("min_date"),
                                  pl.col("date_utc").max().alias("max_date"), pl.len().alias("n")).rows())
dd = hb.group_by(["date_utc", "build_num"]).len().sort(["date_utc", "build_num"])
print("dates with both builds present:",
      dd.group_by("date_utc").agg(pl.col("build_num").n_unique().alias("nb")).filter(pl.col("nb") == 2)
        .sort("date_utc")["date_utc"].to_list())
print("per-date (build: n):")
piv = dd.with_columns(pl.col("len").cast(pl.Int64)).pivot(on="build_num", index="date_utc", values="len").sort("date_utc")
print(piv)
print("\nPATH day vs match_date day (matches where the two agree):")
Hp = pl.scan_parquet(sorted(glob.glob(str(BASE / "header" / "dataframe_*")))).select(["key", "match_date"]).collect()
Hp = Hp.with_columns(pl.col("key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day_path"),
                     pl.col("match_date").str.slice(0, 10).alias("date_utc"))
print("path day == match_date day (day-of-month):",
      Hp.filter(pl.col("day_path") == pl.col("date_utc").str.slice(8, 2)).height, "of", Hp.height)
print("path day x match-date month:")
print(Hp.with_columns(pl.col("date_utc").str.slice(0, 7).alias("ym")).group_by(["day_path", "ym"]).len()
        .sort(["day_path", "ym"]).rows())
