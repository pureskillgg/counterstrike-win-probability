"""verify D9-9, step 9: header totals vs round_end counts and label ranges, by build (read-only)."""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(16)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\D9_9verify")
J = pl.read_parquet(SCR / "round_score_join.parquet")
H = pl.scan_parquet(sorted(glob.glob(str(BASE / "header" / "dataframe_*")))).select(
    ["key", "build_num", "t_starters_score_final", "ct_starters_score_final", "match_date"]).collect().rename({"key": "match_key"})
M = J.group_by("match_key").agg(pl.len().alias("n_re"), pl.col("round").min().alias("mn"),
                                pl.col("round").max().alias("mx")).join(H, on="match_key", how="left")
M = M.with_columns((pl.col("t_starters_score_final") + pl.col("ct_starters_score_final")).alias("tot"),
                   pl.max_horizontal("t_starters_score_final", "ct_starters_score_final").alias("mx_score"))
M = M.with_columns((pl.col("tot") - pl.col("n_re")).alias("tot_minus_n"))
print("== header total minus round_end count, by build")
print(M.group_by(["build_num", "tot_minus_n"]).len().sort(["build_num", "tot_minus_n"]).head(30))
print("\n== max(starters score) distribution by build (top 12)")
print(M.group_by(["build_num", "mx_score"]).len().sort(["build_num", "mx_score"]).filter(pl.col("mx_score") >= 12))
print("\n== min label by build:", M.group_by(["build_num", "mn"]).len().sort(["build_num", "mn"]).rows())
print("\n== example 9842 with 14-7 (first five):")
print(M.filter((pl.col("build_num") == 9842) & (pl.col("mx_score") == 14)).head(5))
print("\n== per-build: n_re distribution (top):", M.group_by(["build_num", "n_re"]).len().sort(["build_num", "n_re"]).head(12).rows())
