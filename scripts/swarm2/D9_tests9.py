"""D9 probe 9: characterise the partial round-9 inversion on build 9961 (non-bomb rounds).

For each non-bomb round scored by death consistency, classify: ok_recorded, ok_flipped.
Round-9 inversion candidates: recorded inconsistent but flipped consistent.
Checks: distribution by day, and whether the matches that invert at round 9 also invert at round 13.
ASSUMPTION: same consistency rule as D9_tests4.py/D9_tests8.py.
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
SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\D9")
R = pl.read_parquet(SCR / "D9_round.parquet")
M = pl.read_parquet(SCR / "D9_match.parquet")


def files_of(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


def scan(name, want):
    fd = BASE / f"basic_{name}{SUF}"
    lf = pl.scan_parquet(files_of(fd))
    have = set(lf.collect_schema().names())
    return lf.select([c for c in want if c in have]).collect()


DE = scan("player_death", ["match_key", "round", "player_id", "player_team_code"])
DEu = DE.unique(subset=["match_key", "round", "player_id"], keep="first")
dead = (DEu.group_by(["match_key", "round"])
        .agg((pl.col("player_team_code") == 2).sum().alias("T_dead"),
             (pl.col("player_team_code") == 3).sum().alias("CT_dead")))
BS = scan("bomb_state", ["match_key", "round", "event_type"])
bomb_rounds = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
               .select(["match_key", "round"]).unique().with_columns(pl.lit(True).alias("bomb_any")))
T = (R.join(M.select(["match_key", "build_num", "day"]), on="match_key", how="left", suffix="_m")
       .join(dead, on=["match_key", "round"], how="left")
       .join(bomb_rounds, on=["match_key", "round"], how="left")
       .with_columns(pl.col("bomb_any").fill_null(False), pl.col("T_dead").fill_null(0),
                     pl.col("CT_dead").fill_null(0))
       .filter(~pl.col("bomb_any") & pl.col("n_CT").is_not_null() & (pl.col("build_num") == 9961)
               & pl.col("round").is_in([8, 9, 10, 13]) & (pl.col("player_count") == 20)))
T = T.with_columns(
    pl.when(pl.col("winner_team_code") == 2).then(pl.col("CT_dead") >= pl.col("n_CT"))
      .otherwise(pl.col("T_dead") >= pl.col("n_T")).alias("ok_recorded"),
    pl.when(pl.col("winner_team_code") == 2).then(pl.col("T_dead") >= pl.col("n_T"))
      .otherwise(pl.col("CT_dead") >= pl.col("n_CT")).alias("ok_flipped"))
T = T.with_columns(pl.when(pl.col("ok_recorded")).then(pl.lit("recorded_ok"))
                   .when(pl.col("ok_flipped")).then(pl.lit("flip_ok"))
                   .otherwise(pl.lit("neither")).alias("cls"))
print("== non-bomb pc20 build-9961 rounds 8, 9, 10, 13: class by round")
print(T.group_by(["round", "cls"]).len().sort(["round", "cls"]))
r9 = T.filter(pl.col("round") == 9)
print("\n== round 9: flip_ok rounds by day")
print(r9.filter(pl.col("cls") == "flip_ok").group_by("day").len().sort("day").rows())
print("   round 9 totals by day:", r9.group_by("day").len().sort("day").rows())
mset = set(r9.filter(pl.col("cls") == "flip_ok")["match_key"].to_list())
r13 = T.filter(pl.col("round") == 13)
r13_map = {k: c for k, c in zip(r13["match_key"].to_list(), r13["cls"].to_list())}
print("\n   matches with round-9 flip_ok: %d | of those with a non-bomb round 13 scored: %d"
      % (len(mset), sum(1 for k in mset if k in r13_map)))
print("   of those, round 13 class:", sorted(collections_counter := {}) if False else
      [ (c, sum(1 for k in mset if r13_map.get(k) == c)) for c in ["recorded_ok", "flip_ok", "neither"] ])
