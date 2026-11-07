r"""D5-8 verifier analysis: does the round-13 label invert in regime A, and which independent source agrees?

Independent truth sources (none uses the claimed scratchpad tables):
  reason side : win_reason_code -> side (T: 1,9,13,18 ; CT: 7,8,11,12,17)   [engine round-end reason]
  bomb side   : bomb_exploded -> T (code 2), bomb_defused -> CT (code 3)      [bomb_state events]
  score step  : freeze-end (t_score, ct_score) of round r+1 minus round r, clean +1 for one side, 0 for other
Label under test: winner_team_code (2 = T, 3 = CT assumed; checked against reason side in unaffected rounds).
Halftime: first round whose freeze-end phase is 'second' (ASSUMPTION: phase field marks the side swap).
Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\verify_D5_8_analysis.py
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(40)

OUT = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\D5_8v")
R = pl.read_parquet(OUT / "rounds_v.parquet").sort(["match_key", "round"])
M = pl.read_parquet(OUT / "matches_v.parquet")

T_REASON = [1, 9, 13, 18]
CT_REASON = [7, 8, 11, 12, 17]
R = R.with_columns(
    rside=pl.when(pl.col("reason").is_in(T_REASON)).then(2).when(pl.col("reason").is_in(CT_REASON)).then(3)
    .otherwise(None).cast(pl.Int64),
    bside=pl.when(pl.col("bomb_exp") & ~pl.col("bomb_def")).then(2)
    .when(pl.col("bomb_def") & ~pl.col("bomb_exp")).then(3).otherwise(None).cast(pl.Int64),
    grp=pl.when((pl.col("day") >= "12") & (pl.col("regime") == "A")).then(pl.lit("lateA"))
    .when((pl.col("day") >= "12") & (pl.col("regime") == "B")).then(pl.lit("lateB"))
    .when(pl.col("day") <= "07").then(pl.lit("earlyB")).otherwise(pl.lit("other")),
)

# ---- 1. code-to-side mapping check: winner vs reason side, all rounds with a clean reason
print("== 1. winner_team_code vs reason side (all rounds, clean reason):")
print(R.filter(pl.col("rside").is_not_null()).group_by(["rside", "winner"]).len().sort(["rside", "winner"]).rows())
print("   bomb side vs winner (bomb-outcome rounds):")
print(R.filter(pl.col("bside").is_not_null()).group_by(["bside", "winner"]).len().sort(["bside", "winner"]).rows())

# ---- 2. halftime position from freeze-end phase
ht = (R.filter(pl.col("fe_phase") == "second").group_by("match_key").agg(pl.col("round").min().alias("ht_round")))
M2 = M.join(ht, on="match_key", how="left")
print("\n== 2. first 'second'-phase round per match, by group:")
M2 = M2.with_columns(grp=pl.when((pl.col("day") >= "12") & (pl.col("regime") == "A")).then(pl.lit("lateA"))
                     .when((pl.col("day") >= "12") & (pl.col("regime") == "B")).then(pl.lit("lateB"))
                     .when(pl.col("day") <= "07").then(pl.lit("earlyB")).otherwise(pl.lit("other")))
print(M2.group_by(["grp", "ht_round"]).len().sort(["grp", "ht_round"], nulls_last=True).rows())

R = R.join(ht, on="match_key", how="left").with_columns(rel=pl.col("round") - pl.col("ht_round"))

# ---- 3. label vs reason-side and label vs bomb-side disagreement, by group and absolute round
def rate_table(df, col_truth, title, rounds):
    d = df.filter(pl.col(col_truth).is_not_null() & pl.col("round").is_in(rounds))
    t = (d.with_columns(mm=(pl.col("winner") != pl.col(col_truth)).cast(pl.Int64))
         .group_by(["grp", "round"]).agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm"),
                                          (pl.col("mm").mean()).round(3).alias("rate"))
         .sort(["grp", "round"]))
    print(f"\n{title}")
    print(t.rows())

rate_table(R, "rside", "== 3a. label vs REASON side, absolute round 9..15 (rows: group, round; tuples: n, mm, rate):",
           [9, 10, 11, 12, 13, 14, 15])
rate_table(R, "bside", "== 3b. label vs BOMB side, absolute round 9..15:", [9, 10, 11, 12, 13, 14, 15])
rate_table(R, "rside", "== 3c. label vs REASON side, relative to first 'second' round (rel -2..+2):",
           [])  # placeholder, replaced below
d = R.filter(pl.col("rside").is_not_null() & pl.col("rel").is_between(-2, 2))
t = (d.with_columns(mm=(pl.col("winner") != pl.col("rside")).cast(pl.Int64))
     .group_by(["grp", "rel"]).agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm"), pl.col("mm").mean().round(3).alias("rate"))
     .sort(["grp", "rel"]))
print("\n== 3c. label vs REASON side, relative to first 'second' round:")
print(t.rows())

d = R.filter(pl.col("bside").is_not_null() & pl.col("rel").is_between(-2, 2))
t = (d.with_columns(mm=(pl.col("winner") != pl.col("bside")).cast(pl.Int64))
     .group_by(["grp", "rel"]).agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm"), pl.col("mm").mean().round(3).alias("rate"))
     .sort(["grp", "rel"]))
print("== 3d. label vs BOMB side, relative to first 'second' round:")
print(t.rows())

# ---- 4. planter side on halftime-adjacent rounds (side-based codes: planter must be T = 2)
d = R.filter(pl.col("planter").is_not_null() & pl.col("rel").is_between(-1, 1))
print("\n== 4. planter team code by group and rel (planter should be 2 if codes are side-based):")
print(d.group_by(["grp", "rel", "planter"]).len().sort(["grp", "rel", "planter"]).rows())
R.write_parquet(OUT / "rounds_v2.parquet")
