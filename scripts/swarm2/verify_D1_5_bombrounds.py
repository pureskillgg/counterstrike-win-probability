"""verify_D1_5 bombrounds: bomb-verified winner mismatches by round number and build_num (2024 tome).
Bomb rule (A1): bomb_defused -> CT (3) win; bomb_exploded -> T (2) win. Population: player_count == 20.
Reports, per build and round, rounds with a bomb outcome, and how many have round_end winner != bomb side."""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)

T24 = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF24 = ".2023-12-28,2024-02-25"


def fl(ch):
    return sorted(glob.glob(str(T24 / f"basic_{ch}{SUF24}" / "dataframe_*")))


RE = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "winner_team_code", "player_count"]) for f in fl("round_end")],
               how="vertical_relaxed").unique(["match_key", "round"], keep="first").filter(pl.col("player_count") == 20)
BS = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "event_type"]) for f in fl("bomb_state")], how="vertical_relaxed")
outc = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
        .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
HD0 = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(str(T24 / "header" / "dataframe_*")))], how="vertical_relaxed")
HDm = HD0.select([pl.col("key").alias("match_key"), "build_num"]).unique("match_key")

B = (RE.join(outc, on=["match_key", "round"], how="inner")
     .with_columns(pl.when(pl.col("evs").list.contains("bomb_defused")).then(3).otherwise(2).alias("bomb_side"))
     .with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mm"))
     .join(HDm, on="match_key", how="left"))
print("bomb-verified rounds (pc==20) by build_num and round, mismatches (round_end winner != bomb side):")
print(B.group_by(["build_num", "round"]).agg(pl.len().alias("n_bomb"), pl.col("mm").sum().alias("mm"))
      .filter(pl.col("mm") > 0).sort(["build_num", "round"]))
print("\nbuild_num totals:")
print(B.group_by("build_num").agg(pl.len().alias("n_bomb"), pl.col("mm").sum().alias("mm")))
print("\nbomb-verified mismatches in rounds other than 9, 13, 25, 28:",
      B.filter(pl.col("mm") & ~pl.col("round").is_in([9, 13, 25, 28])).height)
print("bomb-verified mismatches in build 9842:", B.filter(pl.col("mm") & (pl.col("build_num") == 9842)).height)
