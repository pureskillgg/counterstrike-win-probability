"""verify D9-9, step 8: probe round-number gaps in round_end per match (read-only, uses scratch join)."""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40)
SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\D9_9verify")
J = pl.read_parquet(SCR / "round_score_join.parquet")
g = J.group_by("match_key").agg(
    pl.col("round").min().alias("mn"), pl.col("round").max().alias("mx"),
    pl.col("round").n_unique().alias("nu"), pl.len().alias("n"),
    pl.col("build_num").first().alias("build"))
g = g.with_columns((pl.col("mx") - pl.col("nu")).alias("gap"))
print("min round distribution:", g.group_by("mn").len().sort("mn").rows()[:10])
print("gap (max - nunique) distribution:", g.group_by("gap").len().sort("gap").head(15).rows())
print("rows==unique (no duplicate round_end rows):", g.filter(pl.col("n") == pl.col("nu")).height, "of", g.height)
print("by build, matches with gap>0:", g.filter(pl.col("gap") > 0).group_by("build").len().rows())
ex = g.filter(pl.col("gap") > 0).head(3)["match_key"].to_list()
for k in ex:
    print("example", k, sorted(J.filter(pl.col("match_key") == k)["round"].to_list()))
