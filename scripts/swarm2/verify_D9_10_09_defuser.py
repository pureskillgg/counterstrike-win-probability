"""verify_D9_10 step 9: independent side evidence for bomb rounds, and death-label disagreement by round.

  a. defuser side (bomb_defuse.player_team_code) on defused bomb rounds, by label and switch round.
     A defuse is a CT action, so a defuser coded 2 would contradict the rule; a defuser coded 3 supports CT win.
  b. planter identity across the boundary: planter's player_info team_code in round r-1 vs r (is there a physical side change?).
  c. death-label vs player_info disagreement counts by round label (build 9961, pc20).
"""
import collections
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
FS = pl.read_parquet(SCR / "first_switch.parquet")
fsw = dict(zip(FS["match_key"].to_list(), FS["first_switch_round"].to_list()))
R9 = R.filter(pl.col("build_num") == 9961)
R9 = R9.with_columns(pl.col("match_key").replace_strict(fsw, default=None, return_dtype=pl.Int64).alias("swr"))

# a. defuser side
DF = pl.scan_parquet(sorted(glob.glob(str(BASE / f"basic_bomb_defuse{SUF}" / "dataframe_*")))).select(
    ["match_key", "round", "tick", "event_type", "player_team_code"]).collect()
print("bomb_defuse event types:", DF.group_by("event_type").len().sort("event_type").rows())
dsd = (DF.filter(pl.col("player_team_code").is_not_null())
         .group_by(["match_key", "round"]).agg(pl.col("player_team_code").first().alias("defuser_side"),
                                               pl.len().alias("n_defuse_rows")))
Dd = R9.filter(pl.col("defused") & pl.col("round").is_in([8, 9, 10, 12, 13, 14, 24, 25, 26, 27, 28, 29, 30])).join(
    dsd, on=["match_key", "round"], how="left")
print("\na. defused bomb rounds (build 9961): defuser side by label and switch round")
print(Dd.group_by(["round", "swr", "defuser_side"]).len().sort(["round", "swr", "defuser_side"]))

# b. planter identity across boundary
PL = pl.scan_parquet(sorted(glob.glob(str(BASE / f"basic_bomb_state{SUF}" / "dataframe_*")))).select(
    ["match_key", "round", "event_type", "player_id"]).collect().filter(pl.col("event_type") == "bomb_planted").group_by(
    ["match_key", "round"]).agg(pl.col("player_id").first().alias("planter_id"))
PI = pl.scan_parquet(sorted(glob.glob(str(BASE / f"basic_player_info{SUF}" / "dataframe_*")))).select(
    ["match_key", "round", "player_id", "team_code"]).collect()
PIu = PI.unique(subset=["match_key", "round", "player_id"], keep="last")
bp = R9.filter(pl.col("round").is_in([9, 13, 25, 28]) & pl.col("has_bomb_outcome")).select(["match_key", "round", "swr"]).join(
    PL, on=["match_key", "round"], how="left")
bp = bp.join(PIu.rename({"round": "rr", "team_code": "code_r"}), left_on=["match_key", "planter_id", "round"],
             right_on=["match_key", "player_id", "rr"], how="left")
bp = bp.join(PIu.rename({"round": "rq", "team_code": "code_prev"}), left_on=["match_key", "planter_id"],
             right_on=["match_key", "player_id"], how="left")
bp = bp.filter(pl.col("rq") == pl.col("round") - 1).select(["round", "swr", "code_r", "code_prev"])
print("\nb. planter player_info code in round r (code_r) vs r-1 (code_prev), bomb rounds 9/13/25/28:")
print(bp.group_by(["round", "swr", "code_prev", "code_r"]).len().sort(["round", "swr", "code_prev", "code_r"]))

# c. death-label disagreement by label (build 9961)
LB = R9.filter(pl.col("round").is_in([8, 9, 10, 11, 12, 13, 14, 24, 25, 26, 27, 28, 29, 30]))
agg = LB.group_by("round").agg(pl.col("lab_agree").sum().alias("agree"), pl.col("lab_disagree").sum().alias("disagree"))
print("\nc. death-label vs player_info disagreement by label (build 9961, labels 8-14, 24-30):")
print(agg.sort("round"))
print("   total disagree build 9961 all rounds:", int(R9["lab_disagree"].sum()), "| agree:", int(R9["lab_agree"].sum()))
