"""verify D2-5 part 3: explain the claimed exception "reason == T in 1 flipped row" and test round-label shift.

1. Under the D2 reason mapping (T = {1,9,13,17}, CT = {7,8,11,12,18}) vs the corrected mapping, count
   reason == R and reason == T among score-flipped rows (T from freeze-end score deltas, as in verify_D2_5_d).
2. List the surrender-coded rows (17, 18) with R, s_score, s_bomb, to see which side the score says won.
3. Label-shift test on bomb-verified rows: for R(r), compare how often R equals the bomb outcome of r-1 and r+1,
   against how often the round's own bomb outcome equals those neighbours (same rows, same denominator).
4. Peek at the D2 truth table columns (read-only) to see whether its exception row carries a surrender code.

Imports the rebuild from verify_D2_5_d_scoretruth.py (same directory).
Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_D2_5_e_check.py
"""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, r"G:\pureskill\csds-cs2\scripts\swarm2")
import verify_D2_5_d_scoretruth as d  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(40)

ROOT = Path(r"G:\pureskill\csds-cs2")
D2_T = [1, 9, 13, 17]
D2_CT = [7, 8, 11, 12, 18]


def section(t):
    print("\n" + "=" * 100)
    print(t)
    print("=" * 100)


b = d.build("2024")
b = b.with_columns(
    pl.when(pl.col("win_reason_code").is_in(D2_T)).then(pl.lit(2))
    .when(pl.col("win_reason_code").is_in(D2_CT)).then(pl.lit(3))
    .otherwise(pl.lit(None)).cast(pl.Int64).alias("reason_side_D2"))

section("1. D2 mapping vs corrected mapping, 2024 full 5v5 rounds")
print("reason_side (corrected) == R :", b.filter(pl.col("reason_side") == pl.col("winner_team_code")).height, "of", b.height)
print("reason_side_D2         == R :", b.filter(pl.col("reason_side_D2") == pl.col("winner_team_code")).height, "of", b.height)
print("rows where D2 mapping disagrees with R:")
print(b.filter(pl.col("reason_side_D2") != pl.col("winner_team_code"))
      .select(["match_key", "round", "winner_team_code", "win_reason_code", "msg", "s_score", "s_bomb", "T"]))
k = b.filter(pl.col("T").is_not_null())
fl = k.filter(pl.col("winner_team_code") != pl.col("T"))
print(f"\nscore-flipped rows (T known and R != T): {fl.height}")
print("  corrected: reason == R:", fl.filter(pl.col("reason_side") == pl.col("winner_team_code")).height,
      " reason == T:", fl.filter(pl.col("reason_side") == pl.col("T")).height)
print("  D2 map   : reason == R:", fl.filter(pl.col("reason_side_D2") == pl.col("winner_team_code")).height,
      " reason == T:", fl.filter(pl.col("reason_side_D2") == pl.col("T")).height)
print("  D2-map exception rows (reason_D2 == T on a flipped row):")
print(fl.filter(pl.col("reason_side_D2") == pl.col("T"))
      .select(["match_key", "round", "winner_team_code", "win_reason_code", "msg", "s_score", "s_bomb", "T"]))

section("2. Surrender-coded rows (17, 18), 2024 full 5v5")
print(b.filter(pl.col("win_reason_code").is_in([17, 18]))
      .select(["match_key", "round", "winner_team_code", "win_reason_code", "msg", "sx", "sx_nx",
               "s_score", "s_bomb", "T", "reason_side", "reason_side_D2"]))

section("3. Round-label shift test (2024), R(r) vs bomb outcome of r-1 and r+1")
src = b.filter(pl.col("s_bomb").is_not_null()).select(["match_key", "round", "s_bomb", "winner_team_code"])
prev = src.with_columns((pl.col("round") + 1).alias("round")).select(
    ["match_key", "round", pl.col("s_bomb").alias("sb_prev")])
nxt = src.with_columns((pl.col("round") - 1).alias("round")).select(
    ["match_key", "round", pl.col("s_bomb").alias("sb_next")])
j = src.join(prev, on=["match_key", "round"], how="left").join(nxt, on=["match_key", "round"], how="left")
j = j.with_columns((pl.col("winner_team_code") != pl.col("s_bomb")).alias("defect"))
for name, grp in [("defect rows (R != bomb)", j.filter(pl.col("defect"))),
                  ("control rows (R == bomb)", j.filter(~pl.col("defect")))]:
    for nb in ["sb_prev", "sb_next"]:
        g = grp.filter(pl.col(nb).is_not_null())
        print(f"{name}, neighbour {nb}: n={g.height}  R==neighbour {g.filter(pl.col('winner_team_code') == pl.col(nb)).height}"
              f"  own_bomb==neighbour {g.filter(pl.col('s_bomb') == pl.col(nb)).height}")

section("4. D2 truth parquet (read-only): columns and exception check")
try:
    import polars as pl2
    p = ROOT / "logs" / "swarm2" / "D2_truth_2024.parquet"
    dd = pl2.read_parquet(p)
    print("columns:", dd.columns)
    print("rows:", dd.height)
    if "flip" in dd.columns and "win_reason_code" in dd.columns:
        fl2 = dd.filter(pl2.col("flip") == True)  # noqa: E712
        print("D2 flip rows:", fl2.height)
        print("D2 flip rows where reason (D2 map) == T:")
        print(fl2.filter(pl2.col("win_reason_code").is_in(D2_T) & (pl2.col("T") == 2)).select(
            [c for c in ["match_key", "round", "R", "win_reason_code", "T", "sx", "s_score", "s_bomb"] if c in dd.columns]))
except Exception as e:  # diagnostic only
    print("D2 parquet read failed:", repr(e)[:300])
