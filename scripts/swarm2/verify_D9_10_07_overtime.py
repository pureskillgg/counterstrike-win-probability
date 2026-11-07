"""verify_D9_10 step 7: overtime rounds 22-30 in build 9961 matches that reach round 25.

Per match: team-A side per round (player_info, as in step 5), round_state phase at the round's first row,
bomb outcome side and recorded winner where a bomb outcome exists, and the round_end winner.
Question: is there a side change at round 25 (or 28) in player_info, and where does the inversion fall?
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
PI = pl.scan_parquet(sorted(glob.glob(str(BASE / f"basic_player_info{SUF}" / "dataframe_*")))).select(
    ["match_key", "round", "player_id", "team_code"]).collect()
PIu = PI.unique(subset=["match_key", "round", "player_id"], keep="last").sort(["match_key", "player_id", "round"])
first_code = PIu.group_by(["match_key", "player_id"]).agg(pl.col("team_code").first().alias("code_r1"),
                                                           pl.col("round").min().alias("r_first"))
PIa = PIu.join(first_code, on=["match_key", "player_id"], how="left")
mm = PIa.filter(pl.col("round") == pl.col("r_first")).group_by(["match_key", "code_r1"]).len()
topcode = mm.sort(["match_key", "len"], descending=True).group_by("match_key").agg(pl.col("code_r1").first().alias("A_code0"))
PIa = PIa.join(topcode, on="match_key", how="left").with_columns(pl.col("code_r1").eq(pl.col("A_code0")).alias("in_A"))
A_side = (PIa.group_by(["match_key", "round", "in_A"]).agg(pl.col("team_code").mode().first().alias("side"))
            .filter(pl.col("in_A")).select(["match_key", "round", "side"]))
RS = pl.scan_parquet(sorted(glob.glob(str(BASE / f"basic_round_state{SUF}" / "dataframe_*")))).select(
    ["match_key", "round", "tick", "phase"]).collect()
ph = RS.sort(["match_key", "round", "tick"]).group_by(["match_key", "round"]).agg(pl.col("phase").first().alias("phase"))

ot_matches = (R.filter((pl.col("build_num") == 9961) & (pl.col("round") == 25)).select("match_key").unique())
print("build 9961 matches with a round 25 row:", ot_matches.height)
ot_side = A_side.join(ot_matches, on="match_key").filter(pl.col("round").is_between(22, 30))
pv = ot_side.with_columns(pl.col("side").cast(pl.String)).pivot(on="round", index="match_key", values="side")
print("team-A side (2=T,3=CT) rounds 22..30 in OT matches: pattern counts")
cols = [str(c) for c in range(22, 31)]
pcols = [c for c in cols if c in pv.columns]
pat = pv.select(["match_key"] + pcols).with_columns(pl.concat_str([pl.col(c).fill_null("-") for c in pcols], separator=" ").alias("pat"))
print(pat.group_by("pat").len().sort("len", descending=True).head(12))

print("\nphase per round (first row) for OT matches, most common patterns rounds 22-30:")
pph = ph.join(ot_matches, on="match_key").filter(pl.col("round").is_between(22, 30)).with_columns(
    pl.col("round").cast(pl.String).alias("r")).pivot(on="r", index="match_key", values="phase")
pcols2 = [c for c in cols if c in pph.columns]
pp = pph.select(["match_key"] + pcols2).with_columns(pl.concat_str([pl.col(c).fill_null("-") for c in pcols2], separator=" ").alias("pat"))
print(pp.group_by("pat").len().sort("len", descending=True).head(8))

# bomb/winner at 24-30 for these matches
W = R.filter((pl.col("build_num") == 9961) & pl.col("round").is_between(22, 30)).join(ot_matches, on="match_key")
W = W.with_columns(pl.when(pl.col("has_bomb_outcome")).then(
    pl.when(pl.col("defused")).then(pl.lit("B:def->" + "")).otherwise(pl.lit("B:exp"))).otherwise(pl.lit("nb")).alias("bk"))
print("\nper round, recorded winner and bomb outcome in OT matches (counts):")
print(W.group_by(["round", "winner_team_code", "bk"]).len().sort(["round", "winner_team_code", "bk"]))

# planter side in OT bomb rounds
print("\nplanter side (bomb_planted player_team_code) in OT rounds 25-28, build 9961:")
print(W.filter(pl.col("round").is_in([25, 26, 27, 28])).group_by(["round", "planter_side"]).len().sort(["round", "planter_side"]))
