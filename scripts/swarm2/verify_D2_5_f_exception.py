"""verify D2-5 part 4: find the D2 flipped row(s) where the D2 reason mapping equals T, for both T values,
and show the surrender codes and score/bomb/elim truth for them (read-only use of D2 parquet, diagnostic only)."""
import sys
from pathlib import Path
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(30)
ROOT = Path(r"G:\pureskill\csds-cs2")
dd = pl.read_parquet(ROOT / "logs" / "swarm2" / "D2_truth_2024.parquet")
D2_T = [1, 9, 13, 17]
D2_CT = [7, 8, 11, 12, 18]
dd = dd.with_columns(
    pl.when(pl.col("win_reason_code").is_in(D2_T)).then(pl.lit(2))
    .when(pl.col("win_reason_code").is_in(D2_CT)).then(pl.lit(3)).otherwise(pl.lit(None)).cast(pl.Int64).alias("rs_D2"))
fl = dd.filter(pl.col("flip") == True)  # noqa: E712
print("D2 flip rows:", fl.height)
print("D2 flip rows: reason(D2 map) == T, by T value:")
print(fl.with_columns((pl.col("rs_D2") == pl.col("T")).alias("eqT")).group_by(["T", "eqT"]).len().sort(["T", "eqT"]))
print("D2 flip rows by win_reason_code (top):")
print(fl.group_by("win_reason_code").len().sort("win_reason_code"))
print("D2 flip rows with a surrender code (17/18):")
sur = fl.filter(pl.col("win_reason_code").is_in([17, 18]))
print(sur.select(["match_key", "round", "R", "T", "win_reason_code", "rs_D2", "s_score", "s_bomb", "s_elim", "sx", "sx_prev"]))
print("D2 rows with rs_D2 != R:")
print(dd.filter(pl.col("rs_D2") != pl.col("R")).select(["round", "R", "T", "win_reason_code", "rs_D2", "s_score", "s_bomb", "s_elim"]))
