"""verify D9-9, step 11: residual cases (2024-02 tome, read-only, uses the scratch per-round table).

Looks at build 9961:
  (a) round 13: rounds where RE and SB agree (the ~7% residual), split by the scoreboard halftime
      boundary (swap / noswap / tie into round 13), with death-check pass and bomb agreement.
  (b) round 9: RE vs SB disagreement (about 12%), split by the boundary into round 9, with death-check
      pass and bomb agreement.
  (c) an exact reproduction of the claim's numbers (non-bomb rounds, n_CT not null, no SB filter):
      recorded and flipped pass rates for build 9961 round 13, 25/28, other; build 9842 round 13.
Outputs counts and rates only.
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(24)
SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\D9_9verify")
T = pl.read_parquet(SCR / "three_source_round.parquet").sort(["match_key", "round"])

# boundary into round r: compare pre(r) with post(r-1)
T = T.with_columns(
    pl.col("t_post").shift(1).over("match_key").alias("pt"),
    pl.col("ct_post").shift(1).over("match_key").alias("pc"),
    pl.col("round").shift(1).over("match_key").alias("pr"),
)
T = T.with_columns(
    ((pl.col("t_pre") == pl.col("pt")) & (pl.col("ct_pre") == pl.col("pc"))).alias("noswap_ok"),
    ((pl.col("t_pre") == pl.col("pc")) & (pl.col("ct_pre") == pl.col("pt"))).alias("swap_ok"),
).with_columns(
    pl.when(pl.col("pr") != pl.col("round") - 1).then(pl.lit("no_prev"))
      .when(pl.col("noswap_ok") & pl.col("swap_ok")).then(pl.lit("tie"))
      .when(pl.col("noswap_ok")).then(pl.lit("noswap"))
      .when(pl.col("swap_ok")).then(pl.lit("swap"))
      .otherwise(pl.lit("neither")).alias("bnd"))

B = T.filter(pl.col("build_num") == 9961)

print("== (a) build 9961 round 13: RE vs SB agreement, split by boundary into round 13")
r13 = B.filter(pl.col("round") == 13).with_columns((pl.col("RE") == pl.col("SB")).alias("RE_eq_SB"))
print(r13.group_by(["bnd", "RE_eq_SB"]).agg(
    pl.len().alias("n"),
    pl.col("has_bomb").sum().alias("n_bomb"),
    (pl.col("BO") == pl.col("RE")).sum().alias("bomb_eq_RE"),
    (pl.col("BO") == pl.col("SB")).sum().alias("bomb_eq_SB"),
).sort(["bnd", "RE_eq_SB"]))

nb13 = r13.filter(~pl.col("has_bomb"))
print("\nround 13 non-bomb: death-check pass by bnd and RE_eq_SB (n, pass_RE, pass_SB)")
print(nb13.group_by(["bnd", "RE_eq_SB"]).agg(
    pl.len().alias("n"), pl.col("de_RE").mean().round(4).alias("pass_RE"),
    pl.col("de_SB").mean().round(4).alias("pass_SB")).sort(["bnd", "RE_eq_SB"]))

print("\n== (b) build 9961 round 9: RE vs SB agreement, split by boundary into round 9")
r9 = B.filter(pl.col("round") == 9).with_columns((pl.col("RE") == pl.col("SB")).alias("RE_eq_SB"))
print(r9.group_by(["bnd", "RE_eq_SB"]).agg(
    pl.len().alias("n"),
    pl.col("has_bomb").sum().alias("n_bomb"),
    (pl.col("BO") == pl.col("RE")).sum().alias("bomb_eq_RE"),
    (pl.col("BO") == pl.col("SB")).sum().alias("bomb_eq_SB"),
).sort(["bnd", "RE_eq_SB"]))
nb9 = r9.filter(~pl.col("has_bomb"))
print("round 9 non-bomb: death-check pass by bnd and RE_eq_SB")
print(nb9.group_by(["bnd", "RE_eq_SB"]).agg(
    pl.len().alias("n"), pl.col("de_RE").mean().round(4).alias("pass_RE"),
    pl.col("de_SB").mean().round(4).alias("pass_SB")).sort(["bnd", "RE_eq_SB"]))

print("\n== (c) exact claim-style numbers: non-bomb, n_CT not null, no SB filter")
C = T.filter(~pl.col("has_bomb") & pl.col("n_CT").is_not_null() & (pl.col("n_CT") > 0))
C = C.with_columns(
    pl.when(pl.col("round") == 13).then(pl.lit("r13"))
      .when(pl.col("round").is_in([25, 28])).then(pl.lit("r25_28"))
      .otherwise(pl.lit("other")).alias("rc2"))
print(C.group_by(["build_num", "rc2"]).agg(
    pl.len().alias("n"),
    pl.col("de_RE").mean().round(4).alias("recorded_pass"),
    pl.col("de_REflip").mean().round(4).alias("flipped_pass"),
    pl.col("de_SB").mean().round(4).alias("SB_pass"),
).sort(["build_num", "rc2"]))

print("\n== (c2) build 9961 round-13 non-bomb, claim-style: n with SB null")
print(T.filter((pl.col("build_num") == 9961) & (pl.col("round") == 13) & ~pl.col("has_bomb")).select(
    pl.len().alias("n_nonbomb_r13"), pl.col("SB").is_null().sum().alias("SB_null")))
