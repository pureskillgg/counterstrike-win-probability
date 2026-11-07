"""D9 probe 4: independent check of round_end winners using player deaths (no bomb events).

Test (elimination consistency): in a round with NO bomb outcome whose win message is Terrorists_Win,
the round_end winner should be T (2) and every CT (team code 3) should be dead. Consistency =
number of distinct CT deaths >= number of CT players recorded in player_info for that round.
A correct winner passes this; a flipped winner fails it (the surviving T side is the real winner).
Compared across build (9842 vs 9961) and round index class (13, 25, 28, other).

Also: round_state round_freeze_end tick vs round_end ticks (label offset check, same logic as round_start).

ASSUMPTIONS (ASSUMPTION): player_team_code on player_death is the victim's side; n_CT and n_T come from
player_info (deduplicated per player); a round is eligible only if it has no bomb outcome row.
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


DE = scan("player_death", ["match_key", "round", "player_id", "player_team_code"])
DEu = DE.unique(subset=["match_key", "round", "player_id"], keep="first")
dead = (DEu.group_by(["match_key", "round"])
        .agg((pl.col("player_team_code") == 2).sum().alias("T_dead"),
             (pl.col("player_team_code") == 3).sum().alias("CT_dead")))
BS = scan("bomb_state", ["match_key", "round", "event_type"])
bomb_rounds = BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"])).select(
    ["match_key", "round"]).unique().with_columns(pl.lit(True).alias("bomb_any"))

T = (R.join(M.select(["match_key", "build_num"]), on="match_key", how="left")
       .join(dead, on=["match_key", "round"], how="left")
       .join(bomb_rounds, on=["match_key", "round"], how="left")
       .with_columns(pl.col("bomb_any").fill_null(False),
                     pl.col("T_dead").fill_null(0), pl.col("CT_dead").fill_null(0)))
T = T.with_columns(
    pl.when(pl.col("round") == 13).then(pl.lit("r13"))
      .when(pl.col("round").is_in([25, 28])).then(pl.lit("r25_28"))
      .otherwise(pl.lit("other")).alias("rclass"))
E = T.filter(~pl.col("bomb_any") & (pl.col("win_reason_message") == "#SFUI_Notice_Terrorists_Win")
             & (pl.col("n_CT").is_not_null()))
E = E.with_columns(
    (pl.col("CT_dead") >= pl.col("n_CT")).alias("ct_all_dead"),
    (pl.col("winner_team_code") == 2).alias("winner_is_T"))
print("== D9-7 check A: non-bomb 'Terrorists_Win' rounds (T elimination win) by build x round class")
print("   expected: CT all dead (ct_all_dead=True) if round_end winner is correct")
print(E.group_by(["build_num", "rclass", "winner_is_T", "ct_all_dead"]).len()
      .sort(["build_num", "rclass", "winner_is_T", "ct_all_dead"]))
print("   consistency rate (ct_all_dead among winner=T non-bomb Terrorists_Win rounds):")
print(E.filter(pl.col("winner_is_T")).group_by(["build_num", "rclass"])
      .agg(pl.len().alias("n"), pl.col("ct_all_dead").mean().round(4).alias("rate"))
      .sort(["build_num", "rclass"]))

print("\n== D9-7 check B: non-bomb rounds, all message types, winner vs deaths (build x round class)")
F = T.filter(~pl.col("bomb_any") & pl.col("n_CT").is_not_null())
F = F.with_columns(
    pl.when(pl.col("winner_team_code") == 2).then(pl.col("CT_dead") >= pl.col("n_CT"))
      .otherwise(pl.col("T_dead") >= pl.col("n_T")).alias("winner_side_all_opp_dead"))
print(F.group_by(["build_num", "rclass"]).agg(pl.len().alias("n"),
                                              pl.col("winner_side_all_opp_dead").mean().round(4).alias("rate"))
      .sort(["build_num", "rclass"]))

print("\n== D9-7 check C: round_state round_freeze_end tick vs round_end windows (label offset check)")
RS = scan("round_state", ["match_key", "round", "tick", "event_type"]).filter(
    pl.col("event_type") == "round_freeze_end").group_by(["match_key", "round"]).agg(
    pl.col("tick").min().alias("freeze_end_tick"))
RE = R.select(["match_key", "round", "tick"]).rename({"tick": "end_tick"}).unique(subset=["match_key", "round"])
ends_prev = RE.with_columns((pl.col("round") + 1).alias("round")).rename({"end_tick": "end_prev"})
G = RS.join(RE, on=["match_key", "round"], how="inner").join(ends_prev, on=["match_key", "round"], how="left")
print("   rounds with freeze_end and end:", G.height)
print("   freeze_end < own end (inside own round, before end):",
      G.filter(pl.col("freeze_end_tick") < pl.col("end_tick")).height,
      "| freeze_end > own end:", G.filter(pl.col("freeze_end_tick") > pl.col("end_tick")).height,
      "| freeze_end in (prev end, own end]:",
      G.filter((pl.col("freeze_end_tick") > pl.col("end_prev")) & (pl.col("freeze_end_tick") <= pl.col("end_tick"))).height)
