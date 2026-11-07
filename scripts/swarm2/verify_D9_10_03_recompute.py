"""verify_D9_10 step 3: recompute the claimed flip counts from the independent round table.

Claim rule (copied from D9_tests10.py): build 9961, player_count 20, roster present, rounds 8-14 and 24-30.
Non-bomb rounds only. Per round:
  winner 2 (T): R if CT_dead>=n_CT; F if T_dead>=n_T; else N
  winner 3 (CT): R if T_dead>=n_T; F if CT_dead>=n_CT; else N
  bomb outcome present: '-' (not scored)
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)
SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D9_10")
R = pl.read_parquet(SCR / "round_table.parquet")
LABELS = [8, 9, 10, 11, 12, 13, 14, 24, 25, 26, 27, 28, 29, 30]

R = R.with_columns(
    pl.when(pl.col("has_bomb_outcome")).then(pl.lit("-"))
      .when(pl.col("winner_team_code") == 2).then(
          pl.when((pl.col("CT_dead") >= pl.col("n_CT")) & (pl.col("n_CT") > 0)).then(pl.lit("R"))
          .when((pl.col("T_dead") >= pl.col("n_T")) & (pl.col("n_T") > 0)).then(pl.lit("F"))
          .otherwise(pl.lit("N")))
      .otherwise(
          pl.when((pl.col("T_dead") >= pl.col("n_T")) & (pl.col("n_T") > 0)).then(pl.lit("R"))
          .when((pl.col("CT_dead") >= pl.col("n_CT")) & (pl.col("n_CT") > 0)).then(pl.lit("F"))
          .otherwise(pl.lit("N")))
      .alias("c"))
S = R.filter((pl.col("build_num") == 9961) & (pl.col("player_count") == 20) & pl.col("n_CT").is_not_null()
             & pl.col("round").is_in(LABELS))
print("build 9961, pc20, roster present, labels 8-14/24-30: rounds =", S.height, "matches =", S["match_key"].n_unique())
print("   bomb-outcome (not scored) rounds:", S.filter(pl.col("c") == "-").height)
Sc = S.filter(pl.col("c") != "-")
print("   scored (non-bomb) rounds:", Sc.height, "| matches with any scored round:", Sc["match_key"].n_unique())
print("\nper round label (scored = non-bomb with roster; F = flipped; R = consistent; N = neither)")
tab = (Sc.group_by("round").agg(
    pl.len().alias("n"),
    (pl.col("c") == "F").sum().alias("F"),
    (pl.col("c") == "R").sum().alias("R"),
    (pl.col("c") == "N").sum().alias("N")).sort("round"))
print(tab)
print("\nclaim: r13 scored 1778 F 1580 | r9 scored 1873 F 294 | r25 88/88 | r28 84/84")
# matches with flipped rounds (distinct match count per label)
for rr in [9, 13, 25, 28]:
    mf = Sc.filter((pl.col("round") == rr) & (pl.col("c") == "F"))["match_key"].n_unique()
    print(f"   match-level: round {rr} flipped in {mf} matches")

# per-match flip count and pattern check
pm = Sc.group_by("match_key").agg(
    (pl.col("c") == "F").sum().alias("nF"),
    pl.col("round").filter(pl.col("c") == "F").alias("F_rounds"))
print("\nmatches by number of flipped rounds:", pm.group_by("nF").len().sort("nF").rows())
f9 = pm.filter(pl.col("F_rounds").list.contains(9))
print("matches with a flipped round 9:", f9.height, "| of which also flip 13:", f9.filter(pl.col("F_rounds").list.contains(13)).height)
f13 = pm.filter(pl.col("F_rounds").list.contains(13))
print("matches with a flipped round 13:", f13.height, "| of which flip 9:", f13.filter(pl.col("F_rounds").list.contains(9)).height)

# ---- control: same rule on build 9842 (claim says no r13 flips there)
C = R.filter((pl.col("build_num") == 9842) & (pl.col("player_count") == 20) & pl.col("n_CT").is_not_null()
             & pl.col("round").is_in(LABELS) & (pl.col("c") != "-"))
print("\nCONTROL build 9842 scored rounds by label:")
print(C.group_by("round").agg(pl.len().alias("n"), (pl.col("c") == "F").sum().alias("F"),
                              (pl.col("c") == "R").sum().alias("R"), (pl.col("c") == "N").sum().alias("N")).sort("round"))
