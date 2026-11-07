"""D3-5 verifier, step B: round numbering and freeze_end structure in 2024 G: tome (read only)."""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_width_chars(220)
TOME = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"


def rd(ch, cols):
    folder = TOME / ("header" if ch == "header" else f"basic_{ch}{SUF}")
    return pl.concat([pl.read_parquet(f, columns=cols) for f in sorted(glob.glob(str(folder / "dataframe_*")))],
                     how="vertical_relaxed")


RS = rd("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score",
                        "t_score_raw", "ct_score_raw", "phase", "is_warmup"])
print("is_warmup counts:", RS.group_by("is_warmup").len().rows())
print("phase counts (top):", RS.group_by("phase").len().sort("len", descending=True).head(8).rows())
fe = RS.filter(pl.col("event_type") == "round_freeze_end")
print("freeze_end rows:", fe.height, "| by is_warmup:", fe.group_by("is_warmup").len().rows())
print("freeze_end rows per (match, round) distribution:",
      fe.group_by(["match_key", "round"]).len().group_by("len").agg(pl.len().alias("n_pairs")).sort("len").rows())
print("freeze_end round value counts (first 12):",
      fe.group_by("round").len().sort("round").head(12).rows())
print("freeze_end: round==1 rows with is_warmup false:",
      fe.filter((pl.col("round") == 1) & (~pl.col("is_warmup"))).height)
print("freeze_end: round==0 rows:", fe.filter(pl.col("round") == 0).height)
print("matches with any freeze_end:", fe["match_key"].n_unique())

RE = rd("round_end", ["match_key", "round", "winner_team_code", "player_count", "tick"])
print("round_end rows:", RE.height, "| player_count==20:", RE.filter(pl.col("player_count") == 20).height)
print("round_end round value counts (first 6 rounds):", RE.group_by("round").len().sort("round").head(6).rows())
print("matches in round_end:", RE["match_key"].n_unique())

# numbering class per match (round_end)
cls = (RE.group_by("match_key").agg(pl.col("round").min().alias("mn"), pl.col("round").max().alias("mx"),
                                   pl.col("round").n_unique().alias("n"))
       .with_columns((pl.col("mx") - pl.col("mn") + 1 == pl.col("n")).alias("contig")))
print("round_end first-round classes:", cls.group_by("mn").agg(pl.len().alias("n_m")).sort("mn").rows())
print("round_end contiguous (no internal gaps) by min round:", cls.group_by(["mn", "contig"]).agg(pl.len().alias("n_m")).sort(["mn", "contig"]).rows())

# freeze_end round sets vs round_end round sets
fe_n = fe.group_by("match_key").agg(pl.col("round").min().alias("fe_mn"), pl.col("round").max().alias("fe_mx"),
                                   pl.col("round").n_unique().alias("fe_n"))
j = cls.join(fe_n, on="match_key", how="full", coalesce=True)
print("match-level join: round_end only", j.filter(pl.col("n").is_null() | pl.col("fe_n").is_null()).height)
print("fe_mn vs re_mn (count):", j.group_by(["mn", "fe_mn"]).agg(pl.len().alias("n_m")).sort("n_m", descending=True).head(10).rows())
print("fe count == re count:", j.filter((pl.col("n") == pl.col("fe_n"))).height, "of", j.height)
print("fe_mx == re_mx:", j.filter(pl.col("mx") == pl.col("fe_mx")).height)
