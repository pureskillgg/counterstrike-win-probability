"""Locate side swaps in the score timeline and test the cross-swap explanation for round 27.

  Z1. For each boundary r -> r+1 (days 12-25 and days 1-7 for control), share of matches where the freeze-to-freeze
      score change is a valid single-point round result on the SAME side mapping, and on the SWAPPED mapping.
      A swap boundary shows high SWAPPED validity and low SAME validity.
  Z2. Round 27 flagged rows: share with tied scores at freeze 27 (t == ct), and the (t, ct) pairs.
  Z3. Anti-truth check: if scores are tied before a cross-swap boundary, the same-side formula returns the opposite of the
      true winner. Share of round-27 valid S rows where S != truth proxy (W or bomb), by tie status.
ASSUMPTIONS: A16. A valid single-point result: exactly one side rises by 1 and the other by 0.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(16)
ROOT = Path(r"G:\pureskill\csds-cs2")
SCRATCH = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
SUF = ".2023-12-28,2024-02-25"
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
import glob as _g


def read(ch, cols):
    fd = BASE / f"basic_{ch}{SUF}"
    fs = sorted(_g.glob(str(fd / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


rs = read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
fz = rs.filter(pl.col("event_type") == "round_freeze_end").sort(["match_key", "round", "tick"]).unique(
    ["match_key", "round"], keep="first").select(["match_key", "round", "t_score", "ct_score"])
nx = fz.with_columns((pl.col("round") - 1).alias("round")).select(["match_key", "round", pl.col("t_score").alias("tn"),
                                                                    pl.col("ct_score").alias("ctn")])
B = fz.join(nx, on=["match_key", "round"], how="inner").with_columns(
    pl.col("match_key").str.extract(r"csds/\d{4}/\d\d/(\d\d)/", 1).cast(pl.Int32).alias("dd"))


def valid(a, b, c, d):
    da, db = b - a, d - c
    return ((da >= 0) & (da <= 1) & (db >= 0) & (db <= 1) & (da + db == 1))


B = B.with_columns(
    valid(pl.col("t_score"), pl.col("tn"), pl.col("ct_score"), pl.col("ctn")).alias("same_valid"),
    valid(pl.col("ct_score"), pl.col("tn"), pl.col("t_score"), pl.col("ctn")).alias("swap_valid"))
Z1 = B.filter(pl.col("dd") >= 12).group_by("round").agg(pl.len().alias("n"), pl.col("same_valid").mean().alias("P_same_valid"),
                                                       pl.col("swap_valid").mean().alias("P_swap_valid")).sort("round")
print("Z1 days 12-25: validity of same-side vs swapped-side single-point deltas, by boundary round r (r -> r+1)")
print(Z1.filter(pl.col("round").is_in([8, 9, 10, 11, 12, 13, 14, 23, 24, 25, 26, 27, 28, 29, 30])))
Z1b = B.filter(pl.col("dd") <= 7).group_by("round").agg(pl.len().alias("n"), pl.col("same_valid").mean().alias("P_same_valid"),
                                                       pl.col("swap_valid").mean().alias("P_swap_valid")).sort("round")
print("\nZ1 control days 1-7:")
print(Z1b.filter(pl.col("round").is_in([8, 11, 12, 13, 14, 24, 27, 28])))

R = pl.read_parquet(SCRATCH / "verify_D4_6_rounds_2024.parquet")
F27 = R.filter((pl.col("round") == 27) & (pl.col("dd") >= 12) & (pl.col("player_count") == 20)).select(
    ["match_key", "round", "S", "W", "bomb_side", "t_score", "ct_score"])
F27 = F27.with_columns((pl.col("t_score") == pl.col("ct_score")).alias("tied"))
print("\nZ2 round 27 (days 12-25, full): tie share, and S validity")
print(F27.group_by(["tied"]).agg(pl.len().alias("n"), (pl.col("S") != 0).mean().alias("P_S_nonzero"),
                                 ((pl.col("S") != pl.col("W")) & (pl.col("S") != 0)).sum().alias("n_S_ne_W_valid"),
                                 ((pl.col("S") == pl.col("W")) & (pl.col("S") != 0)).sum().alias("n_S_eq_W_valid")))
Z3 = F27.filter(pl.col("S") != 0).group_by("tied").agg(pl.len().alias("n_valid_S"),
                                                      (pl.col("S") == pl.col("W")).mean().alias("P_S_eq_W"))
print("Z3 round 27 valid S: agreement with round_end W by tie status:")
print(Z3)
