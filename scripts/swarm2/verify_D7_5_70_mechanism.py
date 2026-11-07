"""verify_D7_5_70_mechanism: which matches carry the post-change code inversion at rounds 9 and 13.

Format bucket from the match's maximum round (maxr): MR8-type (maxr <= 16), MR12 regulation (17..24),
overtime (>= 25). Compares code vs within-round (no swap inside a round) and code vs DP by bucket and round.
Also lists the post round-13 rows where code agrees with within (the non-inverted minority).
"""
import sys
from pathlib import Path
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(220)
SCRATCH = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
R = pl.read_parquet(str(SCRATCH / "verify_D7_5_rounds.parquet"))
R = R.with_columns(
    pl.when(pl.col("maxr") <= 16).then(pl.lit("MR8-type"))
      .when(pl.col("maxr") <= 24).then(pl.lit("MR12-regulation"))
      .otherwise(pl.lit("overtime")).alias("fmt"))
both = R.filter(pl.col("W_code").is_not_null() & pl.col("W_within").is_not_null())
both = both.with_columns((pl.col("W_code") != pl.col("W_within")).alias("dis"))
print("== post: code != within by format and round (all player counts) ==")
g = both.filter(pl.col("period") == "post").group_by(["fmt", "round"]).agg(
    pl.len().alias("n"), pl.col("dis").sum().alias("dis")).filter(pl.col("dis") > 0).sort(["round", "fmt"])
for row in g.rows():
    print(row)
print("\n== pre: code != within by format and round ==")
g2 = both.filter(pl.col("period") == "pre").group_by(["fmt", "round"]).agg(
    pl.len().alias("n"), pl.col("dis").sum().alias("dis")).filter(pl.col("dis") > 0).sort(["round", "fmt"])
print(g2.rows())
print("\n== post round 13: code agrees with within (non-inverted) vs inverted, by format ==")
r13 = both.filter((pl.col("period") == "post") & (pl.col("round") == 13))
print(r13.group_by(["fmt", "dis"]).len().sort(["fmt", "dis"]).rows())
print("\n== post round 9: inverted by format ==")
r9 = both.filter((pl.col("period") == "post") & (pl.col("round") == 9))
print(r9.group_by(["fmt", "dis"]).len().sort(["fmt", "dis"]).rows())
print("\n== post round 13 agreeing rows: build and map ==")
print(r13.filter(~pl.col("dis")).group_by(["build", "map"]).len().sort("len", descending=True).head(12).rows())
print("\n== post round 13 inverted rows: by day (mdt date) first 12 days ==")
print(r13.with_columns(pl.col("mdt").dt.strftime("%m-%d").alias("day")).group_by(["day", "dis"]).len().sort("day").head(24).rows())
print("\n== post: rounds with bomb outcome and code!=within, by format ==")
print(both.filter((pl.col("period") == "post") & pl.col("dis") & pl.col("W_bomb").is_not_null()).group_by(["fmt", "round"]).len().sort(["round", "fmt"]).rows())
