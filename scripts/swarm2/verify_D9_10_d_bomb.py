"""verify D9-10 (d): game-side checks that do NOT use player death codes.

Two independent checks on build 9961 matches, grouped by the per-match switch format from player_info
(verify_D9_10_c_schedule.py): format 'sw8' (switch at boundary 8->9, shorter matches) vs 'sw12' (MR12 halftime).
  CHECK 1 (planter side): a bomb_planted row must come from a T player (team code 2). Compare the planter's
          player_team_code on bomb_planted against the same player's player_info team_code in that round.
  CHECK 2 (bomb outcome vs label): bomb_exploded => T won (2); bomb_defused => CT won (3). Compare with
          round_end.winner_team_code. Mismatch rate per (format, round).
  CHECK 3 (halftime marker): round_state phase == 'halftime' rounds, to see where the game itself marks halftime.
ASSUMPTION: bomb_defused/bomb_exploded rows define the outcome; if a round has both, defused is used
  (same rule as e2_side_check_r13.py and D9_build.py's bomb side).
ASSUMPTION: format from player_info switch schedule: 'sw8' = first switch at boundary 8, 'sw12' = at 12.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(20)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"


def files_of(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


def scan(name, cols):
    fd = BASE / "header" if name == "header" else BASE / f"basic_{name}{SUF}"
    lf = pl.scan_parquet(files_of(fd))
    have = set(lf.collect_schema().names())
    missing = [c for c in cols if c not in have]
    if missing:
        raise SystemExit(f"[{name}] missing columns {missing}")
    return lf.select(cols).collect()


H = scan("header", ["key", "build_num"]).rename({"key": "match_key"})
B = set(H.filter(pl.col("build_num") == 9961)["match_key"].to_list())
Bl = list(B)

# ---- switch format per match (same rule as verify_D9_10_c_schedule.py)
PI = scan("player_info", ["match_key", "round", "player_id", "team_code"]).filter(pl.col("match_key").is_in(Bl))
PIu = PI.unique(subset=["match_key", "round", "player_id"], keep="last")
a = PIu.rename({"round": "r", "team_code": "tA"})
b = PIu.with_columns((pl.col("round") - 1).alias("round")).rename({"round": "r", "team_code": "tB"})
pairs = a.join(b, on=["match_key", "r", "player_id"], how="inner").with_columns((pl.col("tA") != pl.col("tB")).alias("chg"))
bd = (pairs.group_by(["match_key", "r"]).agg(pl.len().alias("nc"), pl.col("chg").mean().alias("fch"))
      .filter(pl.col("nc") >= 3).with_columns((pl.col("fch") >= 0.5).alias("sw")))
fmt = (bd.filter(pl.col("sw")).group_by("match_key")
       .agg(pl.col("r").min().alias("first_switch")))
fmt = fmt.with_columns(pl.when(pl.col("first_switch") == 8).then(pl.lit("sw8"))
                       .when(pl.col("first_switch") == 12).then(pl.lit("sw12"))
                       .otherwise(pl.lit("sw_other")).alias("fmt"))
allm = pl.DataFrame({"match_key": Bl})
fmt = allm.join(fmt, on="match_key", how="left").with_columns(pl.col("fmt").fill_null("no_switch_seen"))
print("switch format counts (build 9961):", fmt.group_by("fmt").len().sort("fmt").rows())

# ---- CHECK 3: halftime marker from round_state
RS = scan("round_state", ["match_key", "round", "phase"]).filter(pl.col("match_key").is_in(Bl))
ht = (RS.filter(pl.col("phase") == "halftime").group_by("match_key")
      .agg(pl.col("round").min().alias("ht_round_min"), pl.col("round").max().alias("ht_round_max")))
ht = fmt.join(ht, on="match_key", how="left")
print("\nCHECK 3 halftime marker round by format (round_state phase == 'halftime'):")
print(ht.group_by(["fmt", "ht_round_min"]).len().sort(["fmt", "ht_round_min"]).head(40).rows())
print("matches with no halftime rows:", ht.filter(pl.col("ht_round_min").is_null()).height)

# ---- bomb tables
BS = scan("bomb_state", ["match_key", "round", "event_type", "player_team_code", "player_id"]).filter(
    pl.col("match_key").is_in(Bl))
outc = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
        .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
outc = outc.with_columns(
    pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(3)).otherwise(pl.lit(2)).alias("bomb_side"))
plant = (BS.filter(pl.col("event_type") == "bomb_planted")
         .sort("player_id").group_by(["match_key", "round"])
         .agg(pl.col("player_team_code").first().alias("planter_side"), pl.col("player_id").first().alias("planter_id")))

RE = scan("round_end", ["match_key", "round", "winner_team_code", "player_count"]).filter(
    pl.col("match_key").is_in(Bl)).unique(subset=["match_key", "round"], keep="first")

# planter's team in player_info, same round
pl_info = PIu.select(["match_key", "round", "player_id", "team_code"]).rename({"player_id": "planter_id"})
plant_chk = plant.join(pl_info, on=["match_key", "round", "planter_id"], how="left")

J = (RE.join(outc, on=["match_key", "round"], how="inner")
       .join(fmt.select(["match_key", "fmt"]), on="match_key", how="left")
       .with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mismatch")))
J = J.with_columns(pl.when(pl.col("round").is_in([9, 13, 25, 28])).then(pl.col("round").cast(pl.String))
                   .otherwise(pl.lit("other")).alias("rbin"))

print("\nCHECK 2 bomb outcome vs round_end label, by format x round bin (player_count any):")
print(J.group_by(["fmt", "rbin"]).agg(pl.len().alias("n_bomb_rounds"), pl.col("mismatch").sum().alias("mismatch_n"),
                                       pl.col("mismatch").mean().round(4).alias("mismatch_rate"))
      .sort(["fmt", "rbin"]).rows())

print("\nCHECK 2 detail for format sw8, round 9 (bomb rounds): winner vs bomb side")
print(J.filter((pl.col("fmt") == "sw8") & (pl.col("round") == 9)).group_by(["winner_team_code", "bomb_side"])
      .len().sort(["winner_team_code", "bomb_side"]).rows())
print("CHECK 2 detail for format sw12, round 13 (bomb rounds): winner vs bomb side")
print(J.filter((pl.col("fmt") == "sw12") & (pl.col("round") == 13)).group_by(["winner_team_code", "bomb_side"])
      .len().sort(["winner_team_code", "bomb_side"]).rows())

print("\nCHECK 2 mismatch rate for sw12 round 13 by player_count:")
print(J.filter((pl.col("fmt") == "sw12") & (pl.col("round") == 13)).group_by("player_count")
      .agg(pl.len().alias("n"), pl.col("mismatch").mean().round(4).alias("rate")).sort("player_count").rows())

# ---- CHECK 1: planter side
PC = plant_chk.join(fmt.select(["match_key", "fmt"]), on="match_key", how="left")
PC = PC.with_columns(pl.when(pl.col("round").is_in([9, 13, 25, 28])).then(pl.col("round").cast(pl.String))
                     .otherwise(pl.lit("other")).alias("rbin"))
print("\nCHECK 1 planter side (should be 2 = T) by format x round bin: planter_side counts")
print(PC.group_by(["fmt", "rbin", "planter_side"]).len().sort(["fmt", "rbin", "planter_side"]).rows())
print("CHECK 1 planter side vs player_info team of the same planter (same round):")
print(PC.group_by(["fmt", "rbin", "planter_side", "team_code"]).len().sort(["fmt", "rbin", "planter_side", "team_code"]).rows())
