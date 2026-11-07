"""verify_D9_9 step 2c (exploration): which rounds carry a score-stream side swap, per match, by build.

A round carries the swap signature when its straight score delta is not a unit vector and its
swapped-pairing delta is a unit vector. Prints the sorted list of swap rounds per match (compact).
"""
import sys
from collections import Counter
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D9_9")
R = pl.read_parquet(SCR / "verify_D9_9_round.parquet").filter(pl.col("t_after").is_not_null()).sort(["match_key", "round"])


def unit(a, b):
    return ((a == 1) & (b == 0)) | ((a == 0) & (b == 1))


R = R.with_columns(
    unit(pl.col("dT_straight"), pl.col("dCT_straight")).alias("su"),
    unit(pl.col("dT_swapped"), pl.col("dCT_swapped")).alias("wu"),
).with_columns(((~pl.col("su")) & pl.col("wu")).alias("swap_sig"))
per = R.group_by(["build_num", "match_key"]).agg(
    pl.col("round").filter(pl.col("swap_sig")).alias("swap_rounds"),
    pl.col("round").max().alias("max_round"),
    pl.len().alias("n_scored"),
    (pl.col("su") | pl.col("wu")).all().alias("all_unit_either"))
per = per.with_columns(pl.col("swap_rounds").list.len().alias("n_swaps"))
for b in [9842, 9961]:
    g = per.filter(pl.col("build_num") == b)
    print(f"\n== build {b}: matches {g.height}; swap-count distribution:",
          sorted(Counter(g["n_swaps"].to_list()).items()))
    print("   all rounds scored as unit (straight or swapped):", int(g["all_unit_either"].sum()), "of", g.height)
    sigs = Counter(tuple(x) for x in g["swap_rounds"].to_list())
    print("   swap-round patterns (count):")
    for pat, c in sigs.most_common(25):
        print("     ", pat, c)
