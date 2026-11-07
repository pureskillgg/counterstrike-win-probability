r"""Verifier for D5-5, stage 5: last alternatives.

  S1. Label-shift test: does round_end(13) match the winner of an ADJACENT round (12 or 14) by score?
      score-only winner of round 12 uses a swap-aware rule (halftime swap between freeze_end 12 and 13):
        swap:    ct13 == t12 + 1 and t13 == ct12  -> winner 2 (T side in round 12)
                 t13 == ct12 + 1 and ct13 == t12  -> winner 3
        no swap: t13 == t12 + 1 and ct13 == ct12  -> 2 ;  ct13 == ct12 + 1 and t13 == t12 -> 3
      score-only winner of round 14 = naive step 14->15 (no swap at 14->15 in regulation).
  S2. Day trend: label-13 bomb mismatches by (folder day, r1) to test whether the trend is a date effect
      or the r1 regime share.
  S3. Reason determinism at label 13 by r1 (is reason a function of winner in r1-true matches?).

Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\verify_D5_5_06_shift.py
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)

SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D5_5")
W = pl.read_parquet(SCR / "wide.parquet")
R1 = W.filter(pl.col("k") == 1).select(["match_key", pl.col("rw").is_not_null().alias("r1")])
W = W.join(R1, on="match_key", how="left").with_columns(pl.col("r1").fill_null(False))

FE = W.filter(pl.col("fe_tick").is_not_null()).select(["match_key", "k", "t", "ct"])
F12 = FE.filter(pl.col("k") == 12).select(["match_key", pl.col("t").alias("t12"), pl.col("ct").alias("ct12")])
F13 = FE.filter(pl.col("k") == 13).select(["match_key", pl.col("t").alias("t13"), pl.col("ct").alias("ct13")])
F14 = FE.filter(pl.col("k") == 14).select(["match_key", pl.col("t").alias("t14"), pl.col("ct").alias("ct14")])
F15 = FE.filter(pl.col("k") == 15).select(["match_key", pl.col("t").alias("t15"), pl.col("ct").alias("ct15")])
X = F12.join(F13, on="match_key").join(F14, on="match_key").join(F15, on="match_key", how="left")
X = X.with_columns(
    dt14=pl.col("t15") - pl.col("t14"), dct14=pl.col("ct15") - pl.col("ct14"),
    s12_swap=pl.when((pl.col("ct13") == pl.col("t12") + 1) & (pl.col("t13") == pl.col("ct12"))).then(pl.lit(2))
    .when((pl.col("t13") == pl.col("ct12") + 1) & (pl.col("ct13") == pl.col("t12"))).then(pl.lit(3))
    .otherwise(pl.lit(0)),
    s12_noswap=pl.when((pl.col("t13") == pl.col("t12") + 1) & (pl.col("ct13") == pl.col("ct12"))).then(pl.lit(2))
    .when((pl.col("ct13") == pl.col("ct12") + 1) & (pl.col("t13") == pl.col("t12"))).then(pl.lit(3))
    .otherwise(pl.lit(0)),
)
X = X.with_columns(
    s12=pl.when(pl.col("s12_swap") != 0).then(pl.col("s12_swap")).otherwise(pl.col("s12_noswap")),
    s14=pl.when((pl.col("dt14") == 1) & (pl.col("dct14") == 0)).then(pl.lit(2))
    .when((pl.col("dct14") == 1) & (pl.col("dt14") == 0)).then(pl.lit(3)).otherwise(pl.lit(0)),
)
X = X.join(W.filter(pl.col("k") == 13).select(["match_key", "rw", "reason", "bomb_bs", "r1", "step", "day"]),
           on="match_key", how="left")

print("== S1. round_end(13) winner vs score-only winners of adjacent rounds, by r1 (clean only) ==")
S = X.filter(pl.col("rw").is_not_null())
out = []
for name, col in [("s12 (swap-aware, score only)", "s12"), ("s13 naive step", "step"), ("s14 naive step", "s14")]:
    g = (S.filter(pl.col(col) != 0).group_by("r1").agg(
        pl.len().alias("clean_pairs"),
        (pl.col(col) == pl.col("rw")).sum().alias("rw13_eq"))
        .with_columns(pl.lit(name).alias("comparison"),
                      (pl.col("rw13_eq") / pl.col("clean_pairs")).round(3).alias("rate")))
    out.append(g)
print(pl.concat(out).select(["comparison", "r1", "clean_pairs", "rw13_eq", "rate"]).sort(["comparison", "r1"]))

print("\n== S1b. swap-aware winner of round 12 vs round_end(12), by r1 (validates the swap-aware rule) ==")
R12 = W.filter(pl.col("k") == 12).select(["match_key", pl.col("rw").alias("rw12")])
Y = X.join(R12, on="match_key", how="left").filter(pl.col("rw12").is_not_null() & (pl.col("s12") != 0))
print(Y.group_by("r1").agg(pl.len().alias("n"), (pl.col("s12") == pl.col("rw12")).sum().alias("agree"),
                           (pl.col("s12") == pl.col("rw12")).mean().round(4).alias("rate")))

print("\n== S2. label-13 bomb-outcome rows: mismatch by folder day x r1 ==")
B13 = W.filter((pl.col("k") == 13) & pl.col("rw").is_not_null() & pl.col("bomb_bs").is_not_null() & (pl.col("bomb_bs") >= 0))
B13 = B13.with_columns(mm=pl.col("bomb_bs") != pl.col("rw"))
print(B13.group_by(["day", "r1"]).agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm"))
      .sort(["day", "r1"]))
print("\nr1 share among label-13 rows by day (rows with fe13 and rw13):")
L = W.filter((pl.col("k") == 13) & pl.col("fe_tick").is_not_null() & pl.col("rw").is_not_null())
print(L.group_by("day").agg((pl.col("r1").sum() / pl.len()).round(3).alias("r1_share"), pl.len().alias("n"))
      .sort("day"))

print("\n== S3. label 13: round_end reason by winner x r1 (planted-bomb rows marked) ==")
L3 = W.filter((pl.col("k") == 13) & pl.col("rw").is_not_null())
print(L3.with_columns(planted=pl.col("n_plant").is_not_null()).group_by(["r1", "planted", "rw", "reason"])
      .agg(pl.len().alias("n")).sort(["r1", "planted", "rw", "reason"]))
