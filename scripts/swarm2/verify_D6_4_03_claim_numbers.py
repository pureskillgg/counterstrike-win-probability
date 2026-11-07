"""D6-4 verifier, step 3: check each number in the claim against the per-round joins from step 2.
Reads only the scratchpad joins written by verify_D6_4_02_recompute.py.
"""
import sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(12)
SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
j24 = pl.read_parquet(str(SCR / "verify_D6_4_join_2024.parquet"))
j26 = pl.read_parquet(str(SCR / "verify_D6_4_join_2026.parquet"))

def window(day):
    return "d01-07" if day in {"01", "02", "03", "04", "05", "06", "07"} else ("d12-25" if day.isdigit() and 12 <= int(day) <= 25 else "other")

for label, df in [("player_count==20", j24.filter(pl.col("player_count") == 20)), ("all player_count", j24)]:
    r13 = df.filter((pl.col("round") == 13) & pl.col("bomb_side").is_not_null())
    r13 = r13.with_columns(pl.col("day").map_elements(window, return_dtype=pl.Utf8).alias("win"))
    print(f"\n### 2024 round 13, {label}: rows with bomb outcome = {r13.height}")
    print(r13.group_by("win").agg(pl.len().alias("n"),
        (pl.col("bomb_side") != pl.col("winner_team_code")).sum().alias("bomb_ne_roundend")).sort("win"))

# claim core set: round 13, bomb outcome present, player_count 20, days 12-25
base = j24.filter((pl.col("player_count") == 20) & (pl.col("round") == 13) & pl.col("bomb_side").is_not_null()
                  & pl.col("day").map_elements(window, return_dtype=pl.Utf8).eq("d12-25"))
aff = base.filter(pl.col("bomb_side") != pl.col("winner_team_code"))
ctl = base.filter(pl.col("bomb_side") == pl.col("winner_team_code"))
print(f"\n### claim core: affected(bomb!=round_end)={aff.height}  control(bomb==round_end)={ctl.height}")
print("  affected, score side known:", aff.filter(pl.col("score_side").is_not_null()).height)
aff_u = aff.filter(pl.col("score_side").is_not_null())
print("  affected, score==bomb:", (aff_u["score_side"] == aff_u["bomb_side"]).sum(),
      " score!=bomb:", (aff_u["score_side"] != aff_u["bomb_side"]).sum())
print("  affected, round_end differs from score:", (aff_u["winner_team_code"] != aff_u["score_side"]).sum(),
      " round_end differs from bomb:", (aff_u["winner_team_code"] != aff_u["bomb_side"]).sum())
print("  affected, score ambiguous (dT+dCT != 1):", aff.filter(pl.col("score_side").is_null()).height)
print("  affected, dT/dCT pattern counts:")
print(aff.group_by(["dT", "dCT"]).len().sort(["dT", "dCT"]))
ctl_u = ctl.filter(pl.col("score_side").is_not_null())
print(f"  control with score: {ctl_u.height}; score==bomb==round_end: {(ctl_u['score_side']==ctl_u['bomb_side']).sum()}")
print("  control dT/dCT pattern counts:")
print(ctl.group_by(["dT", "dCT"]).len().sort(["dT", "dCT"]))

# unambiguous-only version: score_unambig flag (dT+dCT==1)
aff_s = aff.filter(pl.col("score_unambig") == True)
print(f"\n  affected with score_unambig: {aff_s.height}; of which score==bomb: {(aff_s['score_side']==aff_s['bomb_side']).sum()}")

# other rounds, same check (score vs bomb vs round_end), 2024 player_count 20
oth = j24.filter((pl.col("player_count") == 20) & (pl.col("round") != 13) & pl.col("bomb_side").is_not_null())
oth = oth.filter(pl.col("score_side").is_not_null())
print(f"\n### 2024 other rounds with bomb and score (pc20): {oth.height}")
print("  bomb != round_end:", (oth["bomb_side"] != oth["winner_team_code"]).sum(),
      "  score != bomb:", (oth["score_side"] != oth["bomb_side"]).sum(),
      "  score != round_end:", (oth["score_side"] != oth["winner_team_code"]).sum())
print("  other-round bomb!=round_end by round:")
mm = j24.filter((pl.col("player_count") == 20) & pl.col("bomb_side").is_not_null() & (pl.col("bomb_side") != pl.col("winner_team_code")))
print(mm.group_by("round").len().sort("len", descending=True).head(12))

# 2026 clean control: score vs round_end over all rounds with score side
s26 = j26.filter(pl.col("score_side").is_not_null())
agree = (s26["score_side"] == s26["winner_team_code"]).sum()
dis = (s26["score_side"] != s26["winner_team_code"]).sum()
print(f"\n### 2026 score vs round_end, all rounds with score side: n={s26.height} agree={agree} disagree={dis}")
b26 = j26.filter(pl.col("bomb_side").is_not_null())
print(f"### 2026 bomb vs round_end: n={b26.height} mismatches={(b26['bomb_side']!=b26['winner_team_code']).sum()}")
r13_26 = j26.filter((pl.col("round") == 13) & pl.col("score_side").is_not_null())
print(f"### 2026 round 13 score vs round_end: n={r13_26.height} disagree={(r13_26['score_side']!=r13_26['winner_team_code']).sum()}")
