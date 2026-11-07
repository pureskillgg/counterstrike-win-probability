"""verify_D9_2 2026b: 2026 player_count convention, round-13 bomb mismatch without the pc filter, round-1 presence.

Read-only; projection only.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(20)
T26 = Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds")
SUF = ".2026-09-18,2026-10-08"


def files_of(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


H = pl.scan_parquet(files_of(T26 / "header")).select(["key", "build_num"]).collect().with_columns(
    pl.col("key").alias("match_key"))
RE = pl.scan_parquet(files_of(T26 / f"basic_round_end{SUF}")).select(
    ["match_key", "round", "winner_team_code", "player_count"]).collect()
print("round_end player_count distribution:", RE.group_by("player_count").len().sort("player_count").rows())
BS = pl.scan_parquet(files_of(T26 / f"basic_bomb_state{SUF}")).select(
    ["match_key", "round", "event_type"]).collect()
outc = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
        .group_by(["match_key", "round"]).agg((pl.col("event_type") == "bomb_defused").any().alias("defused"),
                                              (pl.col("event_type") == "bomb_exploded").any().alias("exploded")))
RE = RE.unique(subset=["match_key", "round"], keep="first").join(outc, on=["match_key", "round"], how="left")
RE = RE.with_columns(pl.col("defused").fill_null(False), pl.col("exploded").fill_null(False)).with_columns(
    (pl.col("defused") | pl.col("exploded")).alias("has_out"),
    pl.when(pl.col("defused")).then(pl.lit(3, dtype=pl.Int64)).when(pl.col("exploded")).then(pl.lit(2, dtype=pl.Int64))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("bomb_side")).with_columns(
    pl.when(pl.col("defused") | pl.col("exploded")).then(pl.col("bomb_side") != pl.col("winner_team_code"))
      .otherwise(None).alias("mm"))
RE = RE.join(H.select(["match_key", "build_num"]), on="match_key", how="left")
print("bomb-outcome rows by build (n, mm, round-13 n):")
bo = RE.filter(pl.col("has_out"))
print(bo.group_by("build_num").agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm"),
                                   (pl.col("round") == 13).sum().alias("r13_n")).sort("build_num"))
print("round-13 bomb rows by build and player_count:")
print(bo.filter(pl.col("round") == 13).group_by(["build_num", "player_count"]).len().sort(["build_num", "player_count"]).rows())
mn = RE.group_by(["match_key", "build_num"]).agg(pl.col("round").min().alias("lo"))
print("min round label per match by build:", mn.group_by(["build_num", "lo"]).len().sort(["build_num", "lo"]).rows())
print("\nDONE 2026b")
