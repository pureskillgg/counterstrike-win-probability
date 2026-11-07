"""verify D1-6 (swarm2 verifier), part J: round-coverage structure of the d01-07 and d12-25 2024 matches (all player_counts).
Counts, per day group: matches; rounds present; max round; gaps (max round minus present count); header sum vs round count.
Used only to say whether d01-07 is a population-comparable control for the round-9 check.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(16)

T24 = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF24 = ".2023-12-28,2024-02-25"
DAYS_HI = [f"{d:02d}" for d in range(12, 26)]


def f24(ch):
    return sorted(glob.glob(str(T24 / f"basic_{ch}{SUF24}" / "dataframe_*")))


hf = sorted(glob.glob(str(T24 / "header" / "dataframe_*")))
H = pl.concat([pl.read_parquet(f, columns=["key", "t_starters_score_final", "ct_starters_score_final"]) for f in hf],
              how="vertical_relaxed").select([pl.col("key").alias("match_key"),
                                             (pl.col("t_starters_score_final") + pl.col("ct_starters_score_final")).alias("hsum"),
                                             pl.max_horizontal(["t_starters_score_final", "ct_starters_score_final"]).alias("hmax")]).unique("match_key")
RE = pl.concat([pl.read_parquet(f, columns=["match_key", "round"]) for f in f24("round_end")], how="vertical_relaxed")
RE = RE.unique()
M = RE.group_by("match_key").agg(pl.len().alias("nr"), pl.col("round").max().alias("maxr"), pl.col("round").min().alias("minr"))
M = M.with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"))
M = M.with_columns(pl.when(pl.col("day").is_in(DAYS_HI)).then(pl.lit("d12-25")).otherwise(pl.lit("d01-07")).alias("dg"))
M = M.join(H, on="match_key", how="left")
M = M.with_columns((pl.col("maxr") - pl.col("nr")).alias("gaps"), (pl.col("hsum") - pl.col("nr")).alias("hsum_minus_nr"))
print("matches by dg:", M.group_by("dg").len().sort("dg").to_dicts())
print("min round present:", M.group_by(["dg", "minr"]).len().sort(["dg", "minr"]).to_dicts())
print("gaps (max round - rounds present) by dg, top:")
print(M.group_by(["dg", "gaps"]).len().sort(["dg", "gaps"]).head(20))
print("header total minus round count, by dg (top 12):")
print(M.group_by(["dg", "hsum_minus_nr"]).len().sort("len", descending=True).head(12))
print("header max score distribution by dg:")
print(M.group_by(["dg", "hmax"]).len().sort(["dg", "hmax"]).head(30))
print("round-count per match, mean, by dg:")
print(M.group_by("dg").agg(pl.col("nr").mean().round(2).alias("mean_nr"), pl.col("gaps").mean().round(2).alias("mean_gaps"),
                           pl.len().alias("matches")))
