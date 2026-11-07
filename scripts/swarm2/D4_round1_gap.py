r"""D4 check: 2024 matches without a round-1 round_end row. Missing row, or labels shifted by one?

For each 2024 match: has_round1, number of distinct round_end rows n, max round label, min round label, and the
header total tA + tB. If labels are shifted, the min label would be 0 or the max would exceed the total.
If only round 1 is missing, the labels run 2..(total) and n == total - 1.
Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\D4_round1_gap.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(12)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"


def read_dir(fd, cols):
    fs = sorted(glob.glob(str(fd / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


H = read_dir(BASE / "header", ["key", "t_starters_score_final", "ct_starters_score_final"]).with_columns(
    (pl.col("t_starters_score_final") + pl.col("ct_starters_score_final")).alias("total")).select(["key", "total"])
RE = read_dir(sorted(BASE.glob("basic_round_end.*"))[0], ["match_key", "round"]).unique(["match_key", "round"], keep="first")
g = RE.group_by("match_key").agg(
    pl.len().alias("n"), pl.col("round").min().alias("mn"), pl.col("round").max().alias("mx"),
    (pl.col("round") == 1).any().alias("has1"))
J = g.join(H, left_on="match_key", right_on="key", how="left").with_columns(
    pl.when(pl.col("has1") & (pl.col("n") == pl.col("total"))).then(pl.lit("has1_complete"))
    .when(pl.col("has1")).then(pl.lit("has1_incomplete"))
    .when((pl.col("mx") == pl.col("total")) & (pl.col("mn") == 2)).then(pl.lit("no1_labels_2..total"))
    .when((pl.col("mx") == pl.col("total") - 1) & (pl.col("mn") == 1)).then(pl.lit("no1_labels_1..total-1"))
    .otherwise(pl.lit("no1_other")).alias("pattern"))
print(J.group_by("pattern").len().sort("pattern"))
print("\nmatches without round 1: min label, max label minus total (count):")
nf = J.filter(~pl.col("has1"))
print(nf.with_columns((pl.col("mx") - pl.col("total")).alias("max_minus_total")).group_by(["mn", "max_minus_total"]).len()
      .sort("len", descending=True).head(12))
