"""D9 probe 3: fix the bomb tick-window check and test adjacent-round attribution correctly.

Checks (read-only on the tome; reads D9 scratchpad tables):
  1. Tick scale sanity: round_start tick, round_end tick and bomb outcome tick for a few rounds.
  2. Round denominators for build 9961 (pc20) at each round index with a mismatch.
  3. Adjacency, corrected: is round_end winner of round r equal to the bomb-derived side of round r-1 or
     r+1 (the bomb-derived side comes from the bomb outcome row of that round, where one exists)?
ASSUMPTIONS (ASSUMPTION): bomb-derived side = 3 if bomb_defused else 2 if bomb_exploded (side-check rule);
  bomb outcome row = first outcome row by tick per round; round_start = min tick per round.
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


BS = scan("bomb_state", ["match_key", "round", "tick", "event_type", "player_team_code"])
RST = scan("round_start", ["match_key", "round", "tick"])
RE = R.select(["match_key", "round", "tick", "winner_team_code", "player_count", "win_reason_message"]).rename(
    {"tick": "end_tick"})
RE = RE.unique(subset=["match_key", "round"], keep="first")
bomb_first = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
              .sort("tick").group_by(["match_key", "round"])
              .agg(pl.col("tick").first().alias("bomb_tick"),
                   pl.col("event_type").first().alias("bomb_event")))
plant_first = (BS.filter(pl.col("event_type") == "bomb_planted").sort("tick")
               .group_by(["match_key", "round"]).agg(pl.col("tick").first().alias("plant_tick")))
start = RST.group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("start_tick"))

print("== 1. Tick scale sanity (first 3 matches, rounds 1-3 and 13)")
sample_keys = M.sort("match_key")["match_key"].head(3).to_list()
S = (RE.join(start, on=["match_key", "round"], how="left")
       .join(bomb_first, on=["match_key", "round"], how="left")
       .join(plant_first, on=["match_key", "round"], how="left"))
print(S.filter(pl.col("match_key").is_in(sample_keys) & pl.col("round").is_in([1, 2, 3, 13]))
      .select(["match_key", "round", "start_tick", "plant_tick", "bomb_tick", "end_tick", "bomb_event"])
      .sort(["match_key", "round"]))
print("   median start_tick, plant_tick, bomb_tick, end_tick:",
      S.select([pl.col("start_tick").median(), pl.col("plant_tick").median(),
                pl.col("bomb_tick").median(), pl.col("end_tick").median()]).rows())
print("   round_start rows per round (value counts):",
      RST.group_by(["match_key", "round"]).agg(pl.len().alias("n")).group_by("n")
      .agg(pl.len().alias("rounds")).sort("n").rows())

print("\n   round_start offset test: where does start(r) fall relative to round_end ticks?")
ends = RE.select(["match_key", "round", "end_tick"])
ends_prev = ends.with_columns((pl.col("round") + 1).alias("round")).rename({"end_tick": "end_prev"})
ends_next = ends.with_columns((pl.col("round") - 1).alias("round")).rename({"end_tick": "end_next"})
O = (start.join(ends, on=["match_key", "round"], how="inner")
     .join(ends_prev, on=["match_key", "round"], how="left")
     .join(ends_next, on=["match_key", "round"], how="left"))
n = O.height
print("   rounds with start and end:", n,
      "| start < own end:", O.filter(pl.col("start_tick") < pl.col("end_tick")).height,
      "| start > own end:", O.filter(pl.col("start_tick") > pl.col("end_tick")).height)
print("   start in (prev end, own end]:",
      O.filter((pl.col("start_tick") > pl.col("end_prev")) & (pl.col("start_tick") <= pl.col("end_tick"))).height,
      "| start in (own end, next end]:",
      O.filter((pl.col("start_tick") > pl.col("end_tick")) & (pl.col("start_tick") <= pl.col("end_next"))).height)

print("\n== 2. Build 9961, pc20: rounds with mismatches, denominators per round index")
B = R.join(M.select(["match_key", "build_num"]), on="match_key", how="left").filter(
    pl.col("has_outcome") & (pl.col("player_count") == 20) & (pl.col("build_num") == 9961))
by_round = B.group_by("round").agg(pl.len().alias("n_outcome"), pl.col("mismatch").sum().alias("mm")) \
    .filter(pl.col("mm") > 0).sort("round")
print(by_round)
B0 = R.join(M.select(["match_key", "build_num"]), on="match_key", how="left").filter(
    pl.col("has_outcome") & (pl.col("player_count") == 20) & (pl.col("build_num") == 9842))
print("   build 9842 pc20 outcome rounds with any mismatch:", int(B0["mismatch"].sum()))
print("   build 9961 pc20 rounds 25 and 28 (any outcome) n:",
      B.filter(pl.col("round").is_in([25, 28])).group_by("round").len().sort("round").rows())

print("\n== 3. Adjacency, corrected: round_end winner r vs bomb-derived side of r-1 and r+1")
bside = (bomb_first.with_columns(
    pl.when(pl.col("bomb_event") == "bomb_defused").then(pl.lit(3)).otherwise(pl.lit(2)).alias("bomb_side"))
    .select(["match_key", "round", "bomb_side"]))
w = RE.select(["match_key", "round", "winner_team_code"])
prev_b = bside.with_columns((pl.col("round") + 1).alias("round")).rename({"bomb_side": "bside_prev"})
next_b = bside.with_columns((pl.col("round") - 1).alias("round")).rename({"bomb_side": "bside_next"})
C = (w.join(bside, on=["match_key", "round"], how="inner")
     .join(prev_b, on=["match_key", "round"], how="left")
     .join(next_b, on=["match_key", "round"], how="left")
     .join(M.select(["match_key", "build_num"]), on="match_key", how="left"))
C = C.with_columns((pl.col("bside_prev") == pl.col("winner_team_code")).alias("winner_eq_prev_bomb"),
                   (pl.col("bside_next") == pl.col("winner_team_code")).alias("winner_eq_next_bomb"),
                   (pl.col("bomb_side") == pl.col("winner_team_code")).alias("own_match"))
for rr in [13]:
    for b in [9842, 9961]:
        g = C.filter((pl.col("round") == rr) & (pl.col("build_num") == b))
        print(f"   round {rr} build {b}: n={g.height}, own match={int(g['own_match'].sum())}, "
              f"winner==prev-round bomb side: {int(g['winner_eq_prev_bomb'].fill_null(False).sum())} "
              f"(of {int(g['winner_eq_prev_bomb'].is_not_null().sum())} with prev bomb), "
              f"winner==next-round bomb side: {int(g['winner_eq_next_bomb'].fill_null(False).sum())} "
              f"(of {int(g['winner_eq_next_bomb'].is_not_null().sum())} with next bomb)")

print("\n== 4. Bomb tick window (prev round end, own round end] by mismatch status")
W = (C.join(ends_prev.select(["match_key", "round", "end_prev"]),
            on=["match_key", "round"], how="left")
     .join(bomb_first.select(["match_key", "round", "bomb_tick"]), on=["match_key", "round"], how="left")
     .join(RE.select(["match_key", "round", "end_tick"]), on=["match_key", "round"], how="left"))
W = W.with_columns(
    (pl.col("bomb_side") != pl.col("winner_team_code")).alias("mismatch"),
    pl.when(pl.col("end_prev").is_null()).then(pl.lit(0)).otherwise(pl.col("end_prev")).alias("win_lo"))
W = W.with_columns(
    ((pl.col("bomb_tick") > pl.col("win_lo")) & (pl.col("bomb_tick") <= pl.col("end_tick"))).alias("in_own_window"))
print("   bomb outcome rows by mismatch x in own window (all rounds):")
print(W.group_by(["mismatch", "in_own_window"]).len().sort(["mismatch", "in_own_window"]))
print("   round 13 only:")
print(W.filter(pl.col("round") == 13).group_by(["build_num", "mismatch", "in_own_window"]).len()
      .sort(["build_num", "mismatch", "in_own_window"]))
print("   median of (bomb_tick - end_tick) and (bomb_tick - win_lo) for mismatch vs matched, round 13:")
r13w = W.filter(pl.col("round") == 13)
print(r13w.group_by("mismatch").agg(
    (pl.col("bomb_tick") - pl.col("end_tick")).median().alias("med_bomb_minus_end"),
    (pl.col("bomb_tick") - pl.col("win_lo")).median().alias("med_bomb_minus_prev_end")).rows())
