"""verify_D7_5_120_minround: first round_end round per match by period, and header-total equality counted per match."""
import sys
from pathlib import Path
from collections import Counter
import pickle
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
SCRATCH = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
R = pl.read_parquet(str(SCRATCH / "verify_D7_5_rounds.parquet"))
m = R.group_by("match_key").agg(pl.col("round").min().alias("min_round"), pl.col("round").max().alias("max_round"),
                                pl.col("period").first().alias("period"), pl.col("tF").first().alias("tF"),
                                pl.col("cF").first().alias("cF"))
print("first round_end round by period (matches):", m.group_by(["period", "min_round"]).len().sort(["period", "min_round"]).rows())
mm = m.with_columns((pl.col("tF") + pl.col("cF") == pl.col("max_round")).alias("eq"))
print("matches with tF+cF == last round, by period:", mm.group_by(["period", "eq"]).len().sort(["period", "eq"]).rows())
print("matches total by period:", m.group_by("period").len().sort("period").rows())
