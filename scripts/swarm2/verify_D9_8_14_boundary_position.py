"""verify D9-8 step 14: position of the inconsistent rounds relative to half and overtime boundaries (2024).

Uses the transition table from step 12 (freeze-to-freeze, bake-off swap rule, H inferred from phase).
For each completed round r with H = halftime round and max_rounds = 2H:
  pos = 'H+1 (first round after halftime)'   if r == H + 1
        'OT1 (first overtime round)'          if r == max_rounds + 1
        'after OT swap'                        if r - 1 is an OT swap round (swap_expected(r-1))
        'swap round itself'                    if swap_expected(r)
        'other'                                otherwise
Reports inconsistency rate by position x H x day group, and the round numbers involved.
Output goes to stdout (tee'd by the caller). Scratch parquet only.
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(30)

SCRATCH = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
T = pl.read_parquet(SCRATCH / "verify_D9_8_freeze_transitions.parquet")
print("transitions:", T.height, " H values:", T.group_by("H").agg(pl.len().alias("n")).sort("H").rows())

T = T.with_columns(
    pl.when(pl.col("round") == pl.col("H") + 1).then(pl.lit("H+1 first after halftime"))
    .when(pl.col("round") == pl.col("max_rounds") + 1).then(pl.lit("OT1 first overtime round"))
    .when(pl.col("swap") & (pl.col("round") > pl.col("max_rounds"))).then(pl.lit("OT swap round"))
    .when(pl.col("swap") & (pl.col("round") == pl.col("H"))).then(pl.lit("halftime swap round"))
    .otherwise(pl.lit("other")).alias("pos"))
print("\ninconsistency by position x H x day group:")
print(T.group_by(["dg", "H", "pos"]).agg(pl.len().alias("n"), (~pl.col("consistent")).sum().alias("inconsistent"),
                                         (~pl.col("consistent")).mean().round(3).alias("rate")).sort(["dg", "H", "pos"]))
print("\ninconsistent transitions: round numbers by H and position (d12-25):")
inc = T.filter((~pl.col("consistent")) & (pl.col("dg") == "d12-25"))
print(inc.group_by(["H", "round", "pos"]).agg(pl.len().alias("n")).sort(["H", "round"]))
print("\nH=8 matches in d12-25: transitions by round 8..10 and inconsistency")
h8 = T.filter((pl.col("H") == 8) & (pl.col("dg") == "d12-25") & pl.col("round").is_between(7, 11))
print(h8.group_by("round").agg(pl.len().alias("n"), (~pl.col("consistent")).sum().alias("inconsistent")).sort("round"))
print("\nH=12 matches in d12-25: transitions by round 12..14 and inconsistency")
h12 = T.filter((pl.col("H") == 12) & (pl.col("dg") == "d12-25") & pl.col("round").is_between(11, 15))
print(h12.group_by("round").agg(pl.len().alias("n"), (~pl.col("consistent")).sum().alias("inconsistent")).sort("round"))
