"""verify_D9_10 step 6: is the round-9 subset the same phenomenon as round 13?

  I.  per match (build 9961, pc20): team-A side-change round from player_info (9, 13, 28, none), as in step 5.
  II. round-9 and round-13 status for non-bomb rounds (R/F/N) crossed with the side-change round.
  III. bomb rounds (outcome present) at labels 9 and 13 crossed with the side-change round: bomb side vs recorded winner.
  IV. halftime evidence from round_state: round of the first 'round_announce_last_round_half' event and of the first
      phase == 'second' row, per match, crossed with the side-change round.
ASSUMPTION: team-A side is taken from the first round present for each player (see step 5).
"""
import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(50)
pl.Config.set_tbl_cols(20)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D9_10")
R = pl.read_parquet(SCR / "round_table.parquet")
FS = pl.read_parquet(SCR / "first_switch.parquet")
fsw = dict(zip(FS["match_key"].to_list(), FS["first_switch_round"].to_list()))
print("matches with build 9961 in first_switch table:", len(fsw), "| distribution:", collections.Counter(fsw.values()))

# ---- classify each round (all rounds, build 9961, pc20, roster)
base = R.filter((pl.col("build_num") == 9961) & (pl.col("player_count") == 20) & pl.col("n_CT").is_not_null()
                & pl.col("round").is_in([8, 9, 10, 11, 12, 13, 14, 24, 25, 26, 27, 28, 29, 30]))
base = base.with_columns(
    pl.when(pl.col("has_bomb_outcome")).then(pl.lit("bomb"))
      .when((pl.col("CT_dead") >= pl.col("n_CT")) & (pl.col("winner_team_code") == 2)).then(pl.lit("R"))
      .when((pl.col("T_dead") >= pl.col("n_T")) & (pl.col("winner_team_code") == 2)).then(pl.lit("F"))
      .when((pl.col("T_dead") >= pl.col("n_T")) & (pl.col("winner_team_code") == 3)).then(pl.lit("R"))
      .when((pl.col("CT_dead") >= pl.col("n_CT")) & (pl.col("winner_team_code") == 3)).then(pl.lit("F"))
      .otherwise(pl.lit("N")).alias("c"))
print("rounds in scope:", base.height, "| status counts:", base.group_by("c").len().sort("c").rows())
base = base.with_columns(pl.col("match_key").replace_strict(fsw, default=None, return_dtype=pl.Int64).alias("swr"))

print("\n== I/II. non-bomb status of round 9 and 13 by team-A side-change round")
nb = base.filter(pl.col("c").is_in(["R", "F", "N"]))
for rr in [9, 13]:
    print(f"  round {rr} non-bomb status x switch round:")
    print(nb.filter(pl.col("round") == rr).group_by(["swr", "c"]).len().sort(["swr", "c"]))
print("  matches by switch round (all):", collections.Counter(fsw.values()))

print("\n== III. bomb rounds at 9 and 13 (and 25, 28) by switch round: bomb side vs recorded winner")
bb = base.filter(pl.col("c") == "bomb").with_columns(
    pl.when(pl.col("defused")).then(pl.lit(3)).otherwise(pl.lit(2)).alias("bomb_side"))
bb = bb.with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mismatch"))
print(bb.filter(pl.col("round").is_in([9, 13, 25, 28])).group_by(["round", "swr"]).agg(
    pl.len().alias("n_bomb"), pl.col("mismatch").sum().alias("n_mismatch")).sort(["round", "swr"]))
print("bomb mismatches at round 9 (detail: match, swr, bomb_side, winner, defused, exploded, planter_side):")
print(bb.filter((pl.col("round") == 9) & pl.col("mismatch")).select(
    ["match_key", "swr", "bomb_side", "winner_team_code", "defused", "exploded", "planter_side", "win_reason_message"]))

# ---- IV. halftime evidence from round_state
RS = pl.scan_parquet(sorted(glob.glob(str(BASE / f"basic_round_state{SUF}" / "dataframe_*")))).select(
    ["match_key", "round", "tick", "event_type", "phase"]).collect()
print("\nround_state rows (build 9961 matches only):")
RS9 = RS.filter(pl.col("match_key").is_in(base["match_key"].unique().to_list()))
ann = (RS9.filter(pl.col("event_type").is_in(["round_announce_last_round_half", "round_announce_match_point", "round_announce_match_start"]))
          .group_by(["match_key", "event_type"]).agg(pl.col("round").min().alias("r")))
ann_p = ann.pivot(on="event_type", index="match_key", values="r")
print("announce columns:", ann_p.columns)
ph = (RS9.filter(pl.col("phase") == "second").group_by("match_key").agg(pl.col("round").min().alias("first_second_phase_round")))
ph1 = (RS9.filter(pl.col("phase") == "halftime").group_by("match_key").agg(pl.col("round").min().alias("first_halftime_round")))
M4 = pl.DataFrame({"match_key": list(fsw.keys()), "swr": list(fsw.values())})
M4 = M4.join(ph, on="match_key", how="left").join(ph1, on="match_key", how="left").join(
    ann_p.select([c for c in ann_p.columns if c in ("match_key", "round_announce_last_round_half")]).rename(
        {"round_announce_last_round_half": "last_round_half_r"}) if "round_announce_last_round_half" in ann_p.columns else pl.DataFrame({"match_key": []}),
    on="match_key", how="left")
print("\nswitch round x first 'second' phase round (count):")
print(M4.group_by(["swr", "first_second_phase_round"]).len().sort(["swr", "first_second_phase_round"]).head(30))
if "last_round_half_r" in M4.columns:
    print("switch round x round of round_announce_last_round_half (count):")
    print(M4.group_by(["swr", "last_round_half_r"]).len().sort(["swr", "last_round_half_r"]).head(30))
M4.write_parquet(SCR / "halftime_check.parquet")
