"""verify D7-2 (06): does ledger swap status explain which round-13 rows flip (beyond time or build)?
Also checks whether long matches (>= 14 rounds) have any ledger swap detected, and how the 3,287-row set S covers all flips.
ASSUMPTION: a 'swap at round r-1' is the ledger swap flag in D7_ledger_dp (min-swap DP); its reliability is what is being tested.
"""
import sys
from pathlib import Path
import datetime as dt
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(200)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
LOG = ROOT / "logs" / "swarm2"
SWITCH = dt.datetime(2024, 2, 7, 1, 1, 18)
H = (pl.read_parquet(str(BASE / "header" / "dataframe_00000"), columns=["key", "match_date", "build_num"])
     .rename({"key": "match_key"})
     .with_columns(pl.col("match_date").str.replace("Z$", "").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f", strict=True).alias("mdt_h")))
X = pl.read_parquet(str(LOG / "D7_rounds_with_ledger_dp.parquet"), columns=["match_key", "round", "w_ok", "mdt"]) \
    .join(H.select(["match_key", "build_num"]), on="match_key", how="left")
L = pl.read_parquet(str(LOG / "D7_ledger_dp.parquet"))
sw = L.filter(pl.col("swap") == True).select(["match_key", "round"]).with_columns(pl.lit(True).alias("swap_after_round"))
# swap flag applies to round r (transition r->r+1); first-after-swap round = r+1
X = X.join(sw.with_columns((pl.col("round") + 1).alias("round")), on=["match_key", "round"], how="left") \
     .with_columns(pl.col("swap_after_round").fill_null(False).alias("fas"))
# round-13 rows with a label, cross-tab: build x fas x flip
R13 = X.filter((pl.col("round") == 13) & pl.col("w_ok").is_not_null()).with_columns((~pl.col("w_ok")).alias("flip"))
print("round-13 labelled rows:", R13.height)
print("build x first-after-swap x flip:", R13.group_by(["build_num", "fas", "flip"]).len().sort(["build_num", "fas", "flip"]).rows())
r13_9961_ok_fas = R13.filter((pl.col("build_num") == 9961) & (~pl.col("flip")))
print("9961 round-13 ok rows: fas status:", r13_9961_ok_fas.group_by("fas").len().rows(), "| earliest/latest mdt:",
      r13_9961_ok_fas["mdt"].min(), r13_9961_ok_fas["mdt"].max())
print("9961 round-13 flip rows: fas status:", R13.filter((pl.col("build_num") == 9961) & pl.col("flip")).group_by("fas").len().rows())
print("9961 round-13 ok rows with fas=False, before-switch count:", r13_9961_ok_fas.filter(pl.col("mdt") < SWITCH).height)

# all flips in 2024 (any round) and coverage by S (fas rounds 9/13)
FL = X.filter(pl.col("w_ok").is_not_null() & (~pl.col("w_ok")))
S_mask = (pl.col("fas") & pl.col("round").is_in([9, 13]))
print("\nall flips (any round, any swap):", FL.height, "| flips in S (fas, rounds 9/13):", FL.filter(S_mask).height,
      "| flips outside S:", FL.filter(~S_mask).height)
print("flips outside S by round x build:", FL.filter(~S_mask).group_by(["build_num", "round"]).len().sort(["build_num", "round"]).rows())

# long matches with no ledger swap detected
M = X.group_by("match_key").agg(pl.col("round").max().alias("max_round"), pl.col("fas").any().alias("any_swap"), pl.col("build_num").first())
long_noswap = M.filter((pl.col("max_round") >= 14) & (~pl.col("any_swap")))
print("\nmatches with max round >= 14:", M.filter(pl.col("max_round") >= 14).height,
      "| of which no ledger swap detected:", long_noswap.height, "| by build:", long_noswap.group_by("build_num").len().rows())
print("long no-swap matches (sample):", long_noswap.head(5).with_columns(pl.col("match_key").str.slice(0, 40)))
