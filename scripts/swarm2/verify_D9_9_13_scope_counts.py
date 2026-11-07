"""verify D9-9, step 13: exact scope counts and header reconciliation (2024-02 tome, read-only).

(a) For each build, the number of rounds with a scoreboard winner, the number where round_end (RE)
    disagrees with the scoreboard (SB), by round index, and how many of those sit at a scoreboard
    swap boundary (boundary into round r, from score fields only, no RE).
(b) Header total (t_starters + ct_starters) vs the number of round_end rows per match, by build.
(c) Bomb rounds: RE vs bomb outcome (BO) and SB vs BO, by build, for all round indices.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(70)
pl.Config.set_tbl_cols(16)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\D9_9verify")
T = pl.read_parquet(SCR / "three_source_round.parquet").sort(["match_key", "round"])
T = T.with_columns(
    pl.col("t_post").shift(1).over("match_key").alias("pt"),
    pl.col("ct_post").shift(1).over("match_key").alias("pc"),
    pl.col("round").shift(1).over("match_key").alias("pr"),
).with_columns(
    ((pl.col("t_pre") == pl.col("pt")) & (pl.col("ct_pre") == pl.col("pc"))).alias("noswap_ok"),
    ((pl.col("t_pre") == pl.col("pc")) & (pl.col("ct_pre") == pl.col("pt"))).alias("swap_ok"),
).with_columns(
    pl.when(pl.col("pr") != pl.col("round") - 1).then(pl.lit("no_prev"))
      .when(pl.col("noswap_ok") & pl.col("swap_ok")).then(pl.lit("tie"))
      .when(pl.col("noswap_ok")).then(pl.lit("noswap"))
      .when(pl.col("swap_ok")).then(pl.lit("swap"))
      .otherwise(pl.lit("neither")).alias("bnd"))

S = T.filter(pl.col("SB").is_not_null())
print("== (a) rounds with SB, RE!=SB by round index and boundary (build 9961; build 9842 total mismatches)")
for bld in [9961, 9842]:
    X = S.filter(pl.col("build_num") == bld).with_columns((pl.col("RE") != pl.col("SB")).alias("dis"))
    print(f"build {bld}: rounds with SB {X.height}, RE!=SB {int(X['dis'].sum())}")
    dis = X.filter(pl.col("dis")).group_by(["round", "bnd"]).len().sort(["round", "bnd"])
    print(dis.filter(pl.col("round") <= 30))
    bnd_tot = X.group_by(["round", "bnd"]).agg(pl.len().alias("n"), pl.col("dis").sum().alias("dis_n")).filter(
        (pl.col("bnd") == "swap")).sort("round")
    print("swap-boundary rounds by round index (n, RE!=SB):", bnd_tot.filter(pl.col("round") <= 30).rows())

print("\n== (c) bomb rounds (BO present), by build x round index class, RE vs BO and SB vs BO")
Bo = T.filter(pl.col("has_bomb") & pl.col("SB").is_not_null())
print(Bo.with_columns(
    pl.when(pl.col("round") == 13).then(pl.lit("r13")).when(pl.col("round").is_in([25, 28])).then(pl.lit("r25_28"))
      .otherwise(pl.lit("other")).alias("rc")
).group_by(["build_num", "rc"]).agg(
    pl.len().alias("n"),
    (pl.col("RE") == pl.col("BO")).mean().round(4).alias("RE_eq_BO"),
    (pl.col("SB") == pl.col("BO")).mean().round(4).alias("SB_eq_BO"),
).sort(["build_num", "rc"]))

print("\n== (b) header total vs round_end row count per match, by build")
H = pl.scan_parquet(sorted(glob.glob(str(BASE / "header" / "dataframe_*")))).select(
    ["key", "build_num", "t_starters_score_final", "ct_starters_score_final"]).collect().rename({"key": "match_key"})
M = T.group_by("match_key").agg(pl.len().alias("n_re")).join(H, on="match_key", how="left")
M = M.with_columns((pl.col("t_starters_score_final") + pl.col("ct_starters_score_final") - pl.col("n_re")).alias("d"))
print(M.group_by("build_num").agg(
    pl.len().alias("matches"),
    (pl.col("d") == 0).sum().alias("total_equals_rounds"),
    (pl.col("d") > 0).sum().alias("header_total_greater"),
    (pl.col("d") < 0).sum().alias("header_total_less"),
    pl.col("d").median().alias("median_diff"),
).sort("build_num"))
