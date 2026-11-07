"""D7-14: build_num (header) versus the ledger flip status and the lag regime, per match.
Flip status: first-after-swap rounds (9/13/28) from the min-swap ledger (D7_8/D7_11). Lag regime: bomb-tick minus round_end tick = -429.
"""
import sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(240)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
H = pl.read_parquet(str(BASE / "header" / "dataframe_00000"), columns=["key", "match_date", "build_num"]).rename({"key": "match_key"}) \
    .with_columns(pl.col("match_date").str.replace("Z$", "").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f", strict=False).alias("mdt"))
X = pl.read_parquet(str(ROOT / "logs" / "swarm2" / "D7_rounds_with_ledger_dp.parquet"))
L = pl.read_parquet(str(ROOT / "logs" / "swarm2" / "D7_ledger_dp.parquet"))
fas = L.filter(pl.col("swap") == True).with_columns((pl.col("round") + 1).alias("round")).select(["match_key", "round"]).with_columns(pl.lit(True).alias("fas"))
Y = X.join(fas, on=["match_key", "round"], how="left").with_columns(pl.col("fas").fill_null(False)) \
     .filter(pl.col("w_ok").is_not_null() & pl.col("fas") & pl.col("round").is_in([9, 13]))
Y = Y.join(H.select(["match_key", "build_num"]), on="match_key", how="left")
print("fas rounds 9/13 rows:", Y.height, "| build_num missing:", Y.filter(pl.col("build_num").is_null()).height)
print("\nbuild_num x winner-agrees-with-ledger (fas rounds 9/13):")
print(Y.group_by(["build_num", "w_ok"]).len().sort(["build_num", "w_ok"]).rows())
print("\nbuild 9842 fas rows by time (after 2024-02-07 00:00):")
print(Y.filter((pl.col("build_num") == 9842) & (pl.col("mdt") >= pl.datetime(2024, 2, 7))).select(["mdt", "round", "w_ok", "match_key"]).sort("mdt").with_columns(pl.col("match_key").str.slice(0, 40)))
print("\nbuild 9961 fas rows that AGREE with ledger (should be none if build is the flip marker):")
print(Y.filter((pl.col("build_num") == 9961) & pl.col("w_ok")).height)
# lag regime per match
B = X.filter(pl.col("bomb_side").is_not_null()).with_columns((pl.col("out_tick") - pl.col("tick")).alias("lag")) \
     .group_by("match_key").agg((pl.col("lag") == -429).any().alias("neg429"))
Z = H.join(B, on="match_key", how="inner")
print("\nbuild_num x lag -429 present (matches with bomb rounds):")
print(Z.group_by(["build_num", "neg429"]).len().sort(["build_num", "neg429"]).rows())
