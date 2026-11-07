"""verify D5-6, second pass (independent of other verify_D5_6_* files).

Reads the per-round table written by verify_D5_6_indep_check.py (scratchpad D5_6_indep/rounds.parquet)
and the header final scores. Adds:
  V1  regime B header validation (round_end has no label 1; round 1 taken from fe(2): ASSUMPTION A5,
      scores start 0-0 so fe_t at round 2 == 1 means the T side won round 1)
  V2  residual of the header test after flipping only round 13 in regime A (distribution)
  V3  reverse elimination check at label 13 (opposite side to the declared loser has >=5 victims)
  V4  reason-code mix of label-13 death/round_end disagreements, regime A vs regime B
ASSUMPTIONS: A1 (MR12 swap after round 12), A2 (roster 5), A5 (scores start 0-0 in round 1).
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_6_indep_check2.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(60)

BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\D5_6_indep")


def section(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


R = pl.read_parquet(SCR / "rounds.parquet")
hf = sorted(glob.glob(str(BASE / "header" / "dataframe_*")))[0]
H = (pl.read_parquet(hf, columns=["key", "t_starters_score_final", "ct_starters_score_final"])
     .rename({"key": "match_key"}))
MM = (R.group_by("match_key").agg(
    pl.col("round").max().alias("max_round"), pl.col("round").n_unique().alias("n_rounds"),
    pl.col("r1_end").first().alias("r1_end"), pl.col("regime").first().alias("regime"),
    pl.col("grp").first().alias("grp"))
    .join(H, on="match_key", how="left"))
MM = MM.with_columns(final_sum=pl.col("t_starters_score_final") + pl.col("ct_starters_score_final"))
R = R.join(MM.select(["match_key", "t_starters_score_final", "final_sum"]), on="match_key", how="left")

# team-A win per round under A1 (A starts on T; halftime swap after round 12)
R = R.with_columns(a_as=pl.when(pl.col("round") <= 12).then((pl.col("rw") == 2).cast(pl.Int32))
                   .otherwise((pl.col("rw") == 3).cast(pl.Int32)))
R = R.with_columns(a_flip13=pl.when(pl.col("round") == 13).then(1 - pl.col("a_as")).otherwise(pl.col("a_as")))

# ------------------------------------------------------------------ V1: regime B header validation
section("V1. Regime B header validation (round_end label 1 absent; round 1 from fe(2) per A5)")
r2 = R.filter(pl.col("round") == 2).select(["match_key", pl.col("fe_t").alias("fe_t_r2")])
rB = MM.filter((pl.col("regime") == "B") & (pl.col("max_round") <= 24)
               & (pl.col("n_rounds") == pl.col("max_round") - 1)
               & (pl.col("final_sum") == pl.col("max_round")))
print("regime B matches eligible (labels 2..max contiguous, header sum == max round, <=24):", rB.height,
      " of regime B:", MM.filter(pl.col("regime") == "B").height)
XB = R.filter(pl.col("match_key").is_in(rB["match_key"]) & pl.col("rw").is_not_null())
perB = XB.group_by("match_key").agg(
    pl.col("a_as").sum().alias("a_rounds_2plus"),
    pl.col("a_flip13").sum().alias("a_rounds_2plus_flip13"),
    pl.col("a_as").filter(pl.col("round") == 13).sum().alias("a13"),
    pl.col("a_flip13").filter(pl.col("round") == 13).sum().alias("a13_flip"))
perB = perB.join(r2, on="match_key", how="left").join(
    rB.select(["match_key", "grp", "t_starters_score_final"]), on="match_key", how="left")
perB = perB.with_columns(a1=pl.when(pl.col("fe_t_r2") == 1).then(1).otherwise(0))
perB = perB.with_columns(
    ok_B=(pl.col("a_rounds_2plus") + pl.col("a1") == pl.col("t_starters_score_final")),
    ok_B_flip13=(pl.col("a_rounds_2plus_flip13") + pl.col("a1") == pl.col("t_starters_score_final")))
print(perB.group_by("grp").agg(
    pl.len().alias("matches"), pl.col("ok_B").mean().round(3).alias("ok_as_regimeB"),
    pl.col("ok_B_flip13").mean().round(3).alias("ok_if_r13_flipped_regimeB")).sort("grp"))
print("regime B: a1 distribution (fe_t at round 2 == 1 -> A won round 1):",
      perB.group_by("a1").agg(pl.len().alias("n")).sort("a1").to_dicts())

# ------------------------------------------------------------------ V2: regime A residual after flipping 13
section("V2. Regime A residual after flipping only round 13 (team-A total minus header final)")
rA = MM.filter((pl.col("regime") == "A") & (pl.col("max_round") <= 24) & (pl.col("n_rounds") == pl.col("max_round"))
               & (pl.col("final_sum") == pl.col("max_round")))
print("regime A eligible (labels 1..max contiguous, header sum == max round, <=24):", rA.height,
      " of regime A:", MM.filter(pl.col("regime") == "A").height)
XA = R.filter(pl.col("match_key").is_in(rA["match_key"]) & pl.col("rw").is_not_null())
perA = XA.group_by("match_key").agg(
    pl.col("a_as").sum().alias("tot_as"),
    pl.col("a_flip13").sum().alias("tot_flip13"),
    (pl.col("round") == 13).any().alias("has13"))
perA = perA.join(rA.select(["match_key", "grp", "t_starters_score_final"]), on="match_key", how="left")
perA = perA.with_columns(
    res_as=pl.col("tot_as") - pl.col("t_starters_score_final"),
    res_flip13=pl.col("tot_flip13") - pl.col("t_starters_score_final"))
print("regime A residual distributions (grp, residual: n):")
print(perA.group_by(["grp", "res_as"]).agg(pl.len().alias("n")).sort(["grp", "res_as"]))
print(perA.group_by(["grp", "res_flip13"]).agg(pl.len().alias("n")).sort(["grp", "res_flip13"]))
print(perA.group_by("grp").agg(
    pl.len().alias("matches"), (pl.col("res_as") == 0).mean().round(3).alias("ok_as"),
    (pl.col("res_flip13") == 0).mean().round(3).alias("ok_flip13")).sort("grp"))

# ------------------------------------------------------------------ V3: reverse elimination check at label 13
section("V3. Reverse elimination check: side opposite to the declared loser has >=5 distinct victims")
el = R.filter(pl.col("reason").is_in([8, 9]) & pl.col("rw").is_not_null() & pl.col("nT").is_not_null()).with_columns(
    declared_loser_n=pl.when(pl.col("rw") == 2).then(pl.col("nCT")).otherwise(pl.col("nT")),
    declared_winner_n=pl.when(pl.col("rw") == 2).then(pl.col("nT")).otherwise(pl.col("nCT")))
el = el.with_columns(loser_ge5=(pl.col("declared_loser_n") >= 5), winner_ge5=(pl.col("declared_winner_n") >= 5))
print("reason 8/9 rounds (label 13 vs other labels 9-16), by grp and regime:")
print(el.filter(pl.col("round").is_between(9, 16) & pl.col("grp").is_in(["ctrl_late", "early"])).with_columns(
    lab=pl.when(pl.col("round") == 13).then(pl.lit("13")).otherwise(pl.lit("other"))).group_by(
    ["grp", "regime", "lab"]).agg(
    pl.len().alias("rounds"),
    pl.col("loser_ge5").mean().round(3).alias("declared_loser_has5"),
    pl.col("winner_ge5").mean().round(3).alias("declared_winner_has5")).sort(["grp", "regime", "lab"]))

# ------------------------------------------------------------------ V4: reason mix of label-13 disagreements
section("V4. Label-13 no-bomb disagreements (last victim != round_end), reason-code mix")
nb = R.filter((~pl.col("has_plant")) & pl.col("implied").is_not_null() & pl.col("rw").is_not_null()
              & (pl.col("round") == 13))
nb = nb.with_columns(dis=(pl.col("implied") != pl.col("rw")))
print(nb.group_by(["grp", "regime", "reason", "dis"]).agg(pl.len().alias("n")).sort(["grp", "regime", "reason", "dis"]))
print("\nDONE")
