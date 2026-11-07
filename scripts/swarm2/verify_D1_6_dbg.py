"""verify D1-6 debug (swarm2 verifier): stage counts for d01-07 in the identity join chain."""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
DAYS_HI = [f"{d:02d}" for d in range(12, 26)]


def fl(ch):
    return sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))


RE = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "winner_team_code", "player_count"])
                for f in fl("round_end")], how="vertical_relaxed")
RE = RE.with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"))
RE = RE.with_columns(pl.when(pl.col("day").is_in(DAYS_HI)).then(pl.lit("d12-25")).otherwise(pl.lit("d01-07")).alias("dg"))
print("stage0 RE d01-07 matches:", RE.filter(pl.col("dg") == "d01-07")["match_key"].n_unique())
W19 = (RE.filter(pl.col("round") <= 9).group_by("match_key").agg(
    pl.col("round").n_unique().alias("nr"),
    (pl.col("winner_team_code") == 2).sum().alias("W_T"),
    (pl.col("winner_team_code") == 3).sum().alias("W_CT")))
print("stage1 W19 rows:", W19.height, " nr value counts:", W19.group_by("nr").len().sort("nr").to_dicts())
meta = RE.select(["match_key", "dg"]).unique()
print("meta by dg:", meta.group_by("dg").len().to_dicts())
print("W19 join meta: d01-07 count:", W19.join(meta, on="match_key", how="left").filter(pl.col("dg") == "d01-07").height)
