"""verify D7-3: is the round_end flip per match (build-level) or per round / map / other covariate?
Truth = score-based winner (round_state t/ct increments), cross-checked against bomb truth. Read-only; prints only.
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(240)
ROOT = Path(r"G:\pureskill\csds-cs2")
SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")

S = pl.read_parquet(str(SCR / "verify_D7_3_score_rounds.parquet"))
S = S.filter(pl.col("has_score")).with_columns(
    (pl.col("winner_team_code") != pl.col("winner_score")).alias("flip"))

print("=== round coverage by build (min/max round, round 1 present?) ===")
cov = S.group_by("build_num").agg(pl.col("round").min().alias("min_r"), pl.col("round").max().alias("max_r"),
                                  pl.col("round").filter(pl.col("round") == 1).len().alias("n_r1"),
                                  pl.len().alias("n_rows"), pl.col("match_key").n_unique().alias("n_matches"))
print(cov)

print("\n=== flip by build x round for ALL rounds with score (flip count, rows) ===")
print(S.group_by(["build_num", "round"]).agg(pl.col("flip").sum().alias("flips"), pl.len().alias("rows"))
      .filter(pl.col("flips") > 0).sort(["build_num", "round"]))

m = S.filter(pl.col("build_num") == 9961)
print("\n=== build 9961: flip rate on round 13 by map ===")
print(m.filter(pl.col("round") == 13).group_by("map_name").agg(pl.len().alias("n"), pl.col("flip").sum().alias("flips"),
                                                              pl.col("flip").mean().alias("rate")).sort("map_name"))
print("\n=== build 9961: flip rate on round 9 by map ===")
print(m.filter(pl.col("round") == 9).group_by("map_name").agg(pl.len().alias("n"), pl.col("flip").sum().alias("flips"),
                                                             pl.col("flip").mean().alias("rate")).sort("map_name"))
print("\n=== build 9842: flip rate on round 13 by map (control) ===")
print(S.filter((pl.col("build_num") == 9842) & (pl.col("round") == 13)).group_by("map_name")
      .agg(pl.len().alias("n"), pl.col("flip").sum().alias("flips")).sort("map_name"))

# per-match coherence of flips among the 'swap' rounds 9/13/25/28 in build 9961
sw = m.filter(pl.col("round").is_in([9, 13, 25, 28]))
pm = sw.group_by(["match_key", "round"]).agg(pl.col("flip").any().alias("f"))
pmw = pm.pivot(on="round", index="match_key", values="f")
print("\nbuild 9961 matches with at least one of rounds 9/13/25/28 present:", pmw.height)
print("columns:", pmw.columns)
cols = [c for c in ["9", "13", "25", "28"] if c in pmw.columns]
print("pattern counts (flip at each of", cols, "):")
print(pmw.group_by(cols).len().sort("len", descending=True))

# does any 9961 round-13 flip depend on the map AND on being 'reached' by a half switch?
print("\n=== build 9961 round-13 non-flip rows: map, has_bomb, player_count, day ===")
nf = m.filter((pl.col("round") == 13) & (~pl.col("flip")))
print(nf.group_by("map_name").len().sort("len", descending=True))
print("non-flip has_bomb:", nf.group_by("has_bomb").len().rows())
print("non-flip reasons:", nf.group_by("win_reason_message").len().rows())

# lag-429 presence per match (from bomb rounds) vs round-13 flip in build 9961
lagm = (S.filter(pl.col("has_bomb")).group_by("match_key")
        .agg((pl.col("lag_bomb_minus_end") == -429).any().alias("has_lag429")))
r13m = m.filter(pl.col("round") == 13).join(lagm, on="match_key", how="left")
print("\nbuild 9961 round-13 rows: flip x has_lag429 (null = no bomb round in match):")
print(r13m.group_by(["flip", "has_lag429"]).len().sort(["flip", "has_lag429"]))

# bomb-truth check for the flipped vs non-flipped 9961 r13 rows with a bomb outcome
bb = m.filter((pl.col("round") == 13) & pl.col("has_bomb"))
print("\nbuild 9961 round-13 rows with bomb outcome: n", bb.height, "| round_end != bomb:",
      bb.filter(pl.col("winner_team_code") != pl.col("bomb_side")).height,
      "| score != bomb:", bb.filter(pl.col("winner_score") != pl.col("bomb_side")).height)
