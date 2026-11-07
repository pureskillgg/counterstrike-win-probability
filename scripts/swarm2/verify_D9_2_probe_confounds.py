"""verify_D9_2 probe: game_build / server_name / rushb_version vs build and calendar date; round-1 presence by build.

Read-only. Header projection; round_end/player_info/round_start projections for label presence.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(50)
SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"


def files_of(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


H = pl.scan_parquet(files_of(BASE / "header")).select(
    ["key", "build_num", "game_directory", "server_name", "rushb_version", "match_date"]).collect()
H = H.with_columns(pl.col("key").alias("match_key"),
                   pl.col("game_directory").str.extract(r"csgo_v(\d+)", 1).alias("game_build"),
                   pl.col("match_date").str.slice(0, 10).alias("date_utc"))
print("== build_num x game_build (matches)")
print(H.group_by(["build_num", "game_build"]).len().sort(["build_num", "game_build"]).rows())
print("== build_num x rushb_version (matches)")
print(H.group_by(["build_num", "rushb_version"]).len().sort(["build_num", "rushb_version"]).rows())
print("== build_num x server_name (matches, top)")
print(H.group_by(["build_num", "server_name"]).len().sort(["build_num", "len"], descending=[False, True]).head(20).rows())
print("== date range by game_build")
print(H.group_by("game_build").agg(pl.col("date_utc").min().alias("min"), pl.col("date_utc").max().alias("max"),
                                  pl.len().alias("n")).sort("game_build").rows())

R = pl.read_parquet(SCR / "verify_D9_2_round.parquet").select(
    ["match_key", "round", "winner_team_code", "build_num", "player_count", "has_out", "mm"])
r13 = R.filter((pl.col("round") == 13) & (pl.col("player_count") == 20) & pl.col("has_out")).join(
    H.select(["match_key", "game_build", "date_utc"]), on="match_key", how="left")
print("\n== round 13 pc20 bomb rows: game_build x mismatch")
print(r13.group_by(["game_build", "mm"]).len().sort(["game_build", "mm"]).rows())

print("\n== round presence by build (round 1 and round 2 in each table)")
for name in ["round_end", "round_start", "bomb_state", "player_info"]:
    fd = BASE / "header" if name == "header" else BASE / f"basic_{name}{SUF}"
    lf = pl.scan_parquet(files_of(fd)).select(["match_key", "round"]).collect()
    lf = lf.join(H.select(["match_key", "build_num"]), on="match_key", how="left")
    per = lf.group_by("build_num").agg(
        pl.col("round").filter(pl.col("round") == 1).len().alias("rows_round1"),
        pl.col("round").filter(pl.col("round") == 2).len().alias("rows_round2"),
        pl.col("match_key").n_unique().alias("matches_with_any_row"))
    print(name, per.sort("build_num").rows())
    if name in ("round_end", "round_start"):
        mins = lf.group_by(["match_key", "build_num"]).agg(pl.col("round").min().alias("lo"))
        print("   min round label per match, by build:",
              mins.group_by(["build_num", "lo"]).len().sort(["build_num", "lo"]).rows())
print("\nDONE confounds")
