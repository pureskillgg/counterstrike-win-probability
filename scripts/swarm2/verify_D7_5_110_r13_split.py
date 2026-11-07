"""verify_D7_5_110_r13_split: round-13 bomb rows split by DP swap flag at 12->13 and by match length (maxr)."""
import sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(220)
SCRATCH = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
R = pl.read_parquet(str(SCRATCH / "verify_D7_5_rounds.parquet")).sort(["match_key", "round"])
R = R.with_columns(pl.col("orient").shift(1).over("match_key").alias("op"), pl.col("round").shift(1).over("match_key").alias("rp"))
R = R.with_columns(((pl.col("orient") != pl.col("op")) & (pl.col("rp") == pl.col("round") - 1)).alias("fas"))
r13 = R.filter((pl.col("period") == "post") & (pl.col("round") == 13) & pl.col("W_bomb").is_not_null() & (pl.col("W_bomb") > 0))
r13 = r13.with_columns((pl.col("maxr") >= 17).alias("long"))
print(r13.group_by(["fas", "long"]).agg(pl.len().alias("n"),
      (pl.col("W_code") == pl.col("W_bomb")).sum().alias("code_eq_bomb"),
      (pl.col("W_within") == pl.col("W_bomb")).sum().alias("within_eq_bomb"),
      pl.col("W_within").is_not_null().sum().alias("n_within"),
      (pl.col("W_dp") == pl.col("W_bomb")).sum().alias("dp_eq_bomb")).sort(["fas", "long"]))
