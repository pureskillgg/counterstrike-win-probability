"""verify_D9_2 probe: round-13 bomb mismatch by calendar match date; the overlap matches; round labels in samples.

Reads scratch parquet from verify_D9_2_core.py plus the header and the bomb_state/round_end/round_start labels.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(120)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(60)
SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"

R = pl.read_parquet(SCR / "verify_D9_2_round.parquet")
H = pl.scan_parquet(sorted(glob.glob(str(BASE / "header" / "dataframe_*")))).select(
    ["key", "match_date", "build_num", "t_starters_score_final", "ct_starters_score_final",
     "rushb_version", "game_directory"]).collect()
H = H.with_columns(pl.col("key").alias("match_key"), pl.col("match_date").str.slice(0, 10).alias("date_utc"),
                   pl.col("match_date").str.slice(11, 8).alias("time_utc"))

print("== round 13 pc20 bomb rows by calendar date and build (n, mm)")
r13 = R.filter((pl.col("player_count") == 20) & pl.col("has_out") & (pl.col("round") == 13)).join(
    H.select(["match_key", "date_utc"]), on="match_key", how="left")
print(r13.group_by(["date_utc", "build_num"]).agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm"))
      .sort(["date_utc", "build_num"]))

print("\n== the two build-9842 matches dated on or after 2024-02-07")
late9842 = H.filter((pl.col("build_num") == 9842) & (pl.col("date_utc") >= "2024-02-07"))
print(late9842.select(["match_key", "match_date", "rushb_version", "game_directory",
                       "t_starters_score_final", "ct_starters_score_final"]))
for mk in late9842["match_key"].to_list():
    rr = R.filter(pl.col("match_key") == mk).sort("round").select(
        ["round", "winner_team_code", "win_reason_message", "player_count", "has_out", "bomb_side", "mm"])
    print("round rows for", mk)
    print(rr)

print("\n== build 9961 matches dated 2024-02-07 (first day) round 13 rows")
d7 = R.filter((pl.col("round") == 13) & (pl.col("player_count") == 20)).join(
    H.select(["match_key", "date_utc", "build_num"]), on="match_key", how="left", suffix="_h")
print(d7.filter(pl.col("date_utc") == "2024-02-07").group_by(["build_num_h", "has_out", "mm"]).len()
      .sort(["build_num_h", "has_out", "mm"]))

print("\n== round labels in sample matches (round_end and bomb outcome)")
samples = (H.filter(pl.col("build_num") == 9842).head(1)["match_key"].to_list()
           + H.filter(pl.col("build_num") == 9961).head(1)["match_key"].to_list())
for mk in samples:
    print("MATCH", mk, H.filter(pl.col("match_key") == mk).select(
        ["build_num", "match_date", "t_starters_score_final", "ct_starters_score_final"]).rows())
    rr = R.filter(pl.col("match_key") == mk).sort("round").select(
        ["round", "tick", "winner_team_code", "win_reason_message", "player_count", "bomb_side", "mm"])
    print(rr)

print("\n== round_start labels vs round_end labels (sample 9842 match)")
RS = pl.scan_parquet(sorted(glob.glob(str(BASE / f"basic_round_start{SUF}" / "dataframe_*")))).select(
    ["match_key", "round", "tick"]).collect()
mk = samples[0]
print("round_start rows:", RS.filter(pl.col("match_key") == mk).sort("round").rows()[:30])
print("round_end  rows (round,tick):", R.filter(pl.col("match_key") == mk).sort("round").select(["round", "tick"]).rows()[:30])
