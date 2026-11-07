"""verify_D9_2 2026: does the 2026 tome contain build 9961 (or 9842) matches, and is round 13 clean there?

Read-only. Header build_num/match_date; round_end and bomb_state with column projection.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(50)
T26 = Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds")


def files_of(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


def one_channel(prefix):
    cands = sorted(glob.glob(str(T26 / f"basic_{prefix}*")))
    return Path(cands[0]) if cands else None


print("channels containing round_end/bomb_state:",
      [p for p in ["round_end", "bomb_state", "header"] if (p == "header" and (T26 / "header").exists())
       or (p != "header" and one_channel(p) is not None)])
H = pl.scan_parquet(files_of(T26 / "header")).select(
    ["key", "build_num", "match_date", "game_directory", "rushb_version"]).collect()
print("2026 header matches:", H.height)
print("build_num counts:", H.group_by("build_num").len().sort("build_num").rows())
print("rushb_version counts:", H.group_by("rushb_version").len().sort("rushb_version").rows())
H = H.with_columns(pl.col("key").alias("match_key"), pl.col("match_date").str.slice(0, 10).alias("date_utc"),
                   pl.col("game_directory").str.extract(r"csgo_v(\d+)", 1).alias("game_build"))
print("date range:", H["date_utc"].min(), H["date_utc"].max())
print("game_build counts:", H.group_by("game_build").len().sort("game_build").rows())

re_dir = one_channel("round_end")
bs_dir = one_channel("bomb_state")
RE = pl.scan_parquet(files_of(re_dir)).select(["match_key", "round", "winner_team_code", "player_count"]).collect()
BS = pl.scan_parquet(files_of(bs_dir)).select(["match_key", "round", "event_type", "player_team_code"]).collect()
outc = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
        .group_by(["match_key", "round"]).agg((pl.col("event_type") == "bomb_defused").any().alias("defused"),
                                              (pl.col("event_type") == "bomb_exploded").any().alias("exploded")))
RE = RE.unique(subset=["match_key", "round"], keep="first").join(outc, on=["match_key", "round"], how="left")
RE = RE.with_columns(pl.col("defused").fill_null(False), pl.col("exploded").fill_null(False))
RE = RE.with_columns(
    (pl.col("defused") | pl.col("exploded")).alias("has_out"),
    pl.when(pl.col("defused")).then(pl.lit(3, dtype=pl.Int64)).when(pl.col("exploded")).then(pl.lit(2, dtype=pl.Int64))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("bomb_side"))
RE = RE.with_columns(pl.when(pl.col("has_out")).then(pl.col("bomb_side") != pl.col("winner_team_code"))
                     .otherwise(None).alias("mm"))
RE = RE.join(H.select(["match_key", "build_num", "date_utc", "game_build"]), on="match_key", how="left")
print("\nround_end rows (2026, unique):", RE.height, "| with bomb outcome:", int(RE["has_out"].sum()))
print("round-13 pc20 bomb rows by build (n, mm):")
r13 = RE.filter((pl.col("round") == 13) & (pl.col("player_count") == 20) & pl.col("has_out"))
print(r13.group_by("build_num").agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")).sort("build_num"))
print("all pc20 bomb rows by build (n, mm):")
pc = RE.filter((pl.col("player_count") == 20) & pl.col("has_out"))
print(pc.group_by("build_num").agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")).sort("build_num"))
print("round 13 pc20 bomb rows by game_build (n, mm):")
print(r13.group_by("game_build").agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")).sort("game_build"))
print("\nDONE 2026")
