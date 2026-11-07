"""verify_D9_10 step 4: strongest alternatives.
  A. build vs date confound (is build 9961 just the defect's date window?)
  B. team-label test: planter side on round-13 bomb rounds (a T-side planter must have side 2)
  C. winner vs bomb outcome by round label and build (independent of death labels)
  D. side switch in player_info per match: which round index shows the switch
ASSUMPTION: player_info.team_code is the player's side in that round (checked in part D by whether it switches).
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D9_10")
R = pl.read_parquet(SCR / "round_table.parquet")

# ---- A. build vs day
print("== A. build by day (matches per day)")
H = pl.read_parquet(SCR / "round_table.parquet").select(["match_key", "build_num", "day"]).unique()
bd = H.group_by(["day", "build_num"]).len().sort(["day", "build_num"])
print(bd.pivot(on="build_num", index="day", values="len").sort("day"))

# ---- B. planter side on round 13 bomb rounds, by build
print("\n== B. planter side (bomb_planted player_team_code) on bomb-outcome rounds")
for lab in [9, 13, 25, 28, 12]:
    sub = R.filter((pl.col("round") == lab) & pl.col("has_bomb_outcome") & pl.col("planter_side").is_not_null())
    print(f"label {lab}: rounds with bomb outcome and a planter row:")
    print(sub.group_by(["build_num", "planter_side"]).len().sort(["build_num", "planter_side"]).rows())

# ---- C. winner vs bomb outcome by label and build
print("\n== C. bomb outcome vs recorded winner (exploded->2, defused->3)")
C = R.filter(pl.col("has_bomb_outcome")).with_columns(
    pl.when(pl.col("defused")).then(pl.lit(3)).otherwise(pl.lit(2)).alias("bomb_side"),
    (pl.col("winner_team_code") != pl.when(pl.col("defused")).then(pl.lit(3)).otherwise(pl.lit(2))).alias("mismatch"))
print("all builds, mismatch by label (labels 8-14, 24-30):")
print(C.filter(pl.col("round").is_in([8, 9, 10, 11, 12, 13, 14, 24, 25, 26, 27, 28, 29, 30]))
        .group_by(["build_num", "round"]).agg(pl.len().alias("n_bomb"), pl.col("mismatch").sum().alias("n_mismatch"))
        .sort(["build_num", "round"]))
print("overall bomb-round mismatch by build (any round):")
print(C.group_by("build_num").agg(pl.len().alias("n"), pl.col("mismatch").sum().alias("mismatch")))

# ---- D. side switch in player_info per match
print("\n== D. player_info side per round: which round index shows the majority side change")
PI = pl.scan_parquet(sorted(glob.glob(str(BASE / f"basic_player_info{SUF}" / "dataframe_*")))).select(
    ["match_key", "round", "player_id", "team_code"]).collect()
PIu = PI.unique(subset=["match_key", "round", "player_id"], keep="last")
# per player per round side; compare each player to their side in the previous round present in data
pr = PIu.sort(["match_key", "player_id", "round"]).with_columns(
    pl.col("team_code").shift(1).over(["match_key", "player_id"]).alias("prev_team"),
    pl.col("round").shift(1).over(["match_key", "player_id"]).alias("prev_round"))
pr = pr.filter(pl.col("prev_team").is_not_null() & (pl.col("round") - pl.col("prev_round") == 1))
chg = pr.with_columns((pl.col("team_code") != pl.col("prev_team")).alias("changed"))
per_round = chg.group_by(["match_key", "round"]).agg(pl.col("changed").mean().alias("frac_changed"),
                                                      pl.len().alias("n_pairs"))
sw = per_round.filter(pl.col("frac_changed") > 0.5)
print("rounds (match-level) where most players changed side vs prior round, by round index:")
print(sw.group_by("round").len().sort("round").rows())
print("count of rounds with frac_changed in (0.2,0.8):", per_round.filter((pl.col("frac_changed") > 0.2) & (pl.col("frac_changed") < 0.8)).height)
# per-match switch round (first round with >50% change) for build 9961
bm = H.select(["match_key", "build_num"])
first_sw = (sw.join(bm, on="match_key", how="left")
              .group_by(["match_key", "build_num"]).agg(pl.col("round").min().alias("first_switch_round")))
print("first switch round distribution, build 9961:", first_sw.filter(pl.col("build_num") == 9961).group_by("first_switch_round").len().sort("first_switch_round").rows())
print("first switch round distribution, build 9842:", first_sw.filter(pl.col("build_num") == 9842).group_by("first_switch_round").len().sort("first_switch_round").rows())
print("matches with no detected switch, build 9961:", bm.filter(pl.col("build_num") == 9961).height - first_sw.filter(pl.col("build_num") == 9961).height,
      "| build 9842:", bm.filter(pl.col("build_num") == 9842).height - first_sw.filter(pl.col("build_num") == 9842).height)
first_sw.write_parquet(SCR / "first_switch.parquet")
