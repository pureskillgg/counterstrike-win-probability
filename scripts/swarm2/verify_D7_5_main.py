"""verify D7-5 step 2: recompute the claimed numbers and test the strongest alternatives.

Input: scratchpad R_all_rounds.parquet written by verify_D7_5_build.py (raw-tome derived, one row per match-round).
Main population: player_count == 20 (ASSUMPTION: 5v5). Ledger = my free min-swap DP (dp_side) and the fixed
halftime schedule (sch_side). Bomb side = exploded -> 2 (T), defused -> 3 (CT), from bomb_state event_type only.
Elimination side = side whose 5 distinct players are all dead inside [freeze_end tick, round_end tick].
ASSUMPTION: onset = 2024-02-07 01:01:18 UTC (the D7 date marker); build_num 9961 is also tested as the marker.
"""
import sys
import glob
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_cols(40)
pl.Config.set_tbl_width_chars(220)

ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D7_5")
ONSET = pl.datetime(2024, 2, 7, 1, 1, 18)
D7_BOUNDARY = pl.datetime(2024, 2, 7, 0, 0, 0)

R0 = pl.read_parquet(str(SCR / "R_all_rounds.parquet"))
print("== population ==")
print("rounds in R:", R0.height, "| matches:", R0["match_key"].n_unique())
print("rounds by player_count (top):", R0.group_by("player_count").len().sort("len", descending=True).head(4).rows())

R = R0.with_columns(
    pl.col("build_num").cast(pl.Utf8).alias("build"),
    pl.when(pl.col("round") == 13).then(pl.lit("r13"))
      .when(pl.col("round") == 25).then(pl.lit("r25"))
      .when((pl.col("round") >= 28) & (((pl.col("round") - 25) % 3) == 0)).then(pl.lit("OT_fas"))
      .otherwise(pl.lit("other")).alias("cls"),
    (pl.col("mdt") >= ONSET).alias("after_onset_ts"),
    (pl.col("mdt") < D7_BOUNDARY).alias("pre_d7"),
)
R = R.with_columns(
    (pl.col("winner_team_code") == pl.col("dp_side")).alias("w_eq_dp"),
    (pl.col("winner_team_code") == pl.col("sch_side")).alias("w_eq_sch"),
    (pl.col("dp_side") == pl.col("sch_side")).alias("dp_eq_sch"),
    (pl.col("winner_team_code") == pl.col("bomb_side")).alias("w_eq_bomb"),
    (pl.col("dp_side") == pl.col("bomb_side")).alias("dp_eq_bomb"),
    (pl.col("sch_side") == pl.col("bomb_side")).alias("sch_eq_bomb"),
    (pl.col("winner_team_code") == pl.col("elim_side")).alias("w_eq_elim"),
    (pl.col("dp_side") == pl.col("elim_side")).alias("dp_eq_elim"),
    (pl.col("sch_side") == pl.col("elim_side")).alias("sch_eq_elim"),
    # side implied by the round_end reason code (T-win reasons 1,9,13,18; CT-win 7,8,11,12,17)
    pl.when(pl.col("win_reason_code").is_in([1, 9, 13, 18])).then(pl.lit(2))
      .when(pl.col("win_reason_code").is_in([7, 8, 11, 12, 17])).then(pl.lit(3))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("reason_side"),
)
R = R.with_columns((pl.col("reason_side") == pl.col("winner_team_code")).alias("reason_eq_winner"))
Q = R.filter(pl.col("player_count") == 20)
print("5v5 rounds:", Q.height, "| 5v5 rounds with dp side:", Q["dp_side"].is_not_null().sum(),
      "| with schedule side:", Q["sch_side"].is_not_null().sum(),
      "| with bomb outcome:", Q["bomb_side"].is_not_null().sum(),
      "| with elimination side:", Q["elim_side"].is_not_null().sum())
print("5v5 matches by build:", Q.group_by("build").agg(pl.col("match_key").n_unique()).sort("build").rows())
print("5v5 rounds by build:", Q.group_by("build").len().sort("build").rows())

print("\n== CLAIM CHECK 1: winner vs dp ledger, D7-style period split (5v5, dp side known) ==")
C1 = Q.filter(pl.col("dp_side").is_not_null())
print(C1.group_by(["pre_d7", "w_eq_dp"]).len().sort(["pre_d7", "w_eq_dp"]).rows(),
      "  (pre_d7 = mdt < 2024-02-07 00:00)")
print(C1.group_by(["build", "w_eq_dp"]).len().sort(["build", "w_eq_dp"]).rows(), "  (by build_num)")
print("pre_d7=True mismatch rounds by round:",
      C1.filter(pl.col("pre_d7") & (pl.col("w_eq_dp") == False)).group_by("round").len().sort("round").rows())

print("\n== CLAIM CHECK 2: first-after-swap (structural: r13, r25, OT swaps) flip rate ==")
fas = C1.filter(pl.col("cls") != "other")
print(fas.group_by(["build", "cls", "w_eq_dp"]).len().sort(["build", "cls", "w_eq_dp"]).rows())
print("non-fas rounds (5v5, dp known):", C1.filter(pl.col("cls") == "other").group_by(["build", "w_eq_dp"]).len().sort(["build", "w_eq_dp"]).rows())

print("\n== CHECK 3: mismatch rate by round number (5v5, dp known), build 9842 vs 9961 ==")
G = (C1.group_by(["round", "build"]).agg((pl.col("w_eq_dp") == False).sum().alias("ne"), pl.len().alias("n"))
     .with_columns(pl.concat_str([pl.col("ne").cast(pl.Utf8), pl.lit("/"), pl.col("n").cast(pl.Utf8)]).alias("ne_over_n")))
Gp = G.pivot(on="build", index="round", values="ne_over_n").sort("round")
print(Gp.filter(pl.col("round") <= 32))

print("\n== CHECK 4: bomb-outcome rounds (5v5), mismatch counts by class and build ==")
B = Q.filter(pl.col("bomb_side").is_not_null())
print(B.group_by(["build", "cls"]).agg(
    pl.len().alias("n_bomb"),
    (pl.col("w_eq_bomb") == False).sum().alias("win_ne_bomb"),
    (pl.col("dp_eq_bomb") == False).sum().alias("dp_ne_bomb"),
    (pl.col("sch_eq_bomb") == False).sum().alias("sch_ne_bomb"),
    pl.col("dp_eq_bomb").is_not_null().sum().alias("n_dp_with_bomb"),
).sort(["build", "cls"]))
print("bomb rounds with both defused and exploded in the same round:",
      Q.filter(pl.col("n_outcome_types") == 2).height)

print("\n== CHECK 5: elimination-determined rounds (5v5), independent of scores and round_end ==")
E = Q.filter(pl.col("elim_side").is_not_null())
print(E.group_by(["build", "cls"]).agg(
    pl.len().alias("n_elim"),
    (pl.col("w_eq_elim") == False).sum().alias("win_ne_elim"),
    (pl.col("dp_eq_elim") == False).sum().alias("dp_ne_elim"),
    (pl.col("sch_eq_elim") == False).sum().alias("sch_ne_elim"),
).sort(["build", "cls"]))

print("\n== CHECK 6: the 499-type mismatch rows (win != dp), build 9961, what do the other sides say ==")
M = C1.filter((pl.col("build") == "9961") & (pl.col("w_eq_dp") == False))
print("mismatch rows (9961, 5v5, dp known):", M.height)
print("  with bomb outcome:", M.filter(pl.col("bomb_side").is_not_null()).height,
      "| bomb_side == dp (ledger):", M.filter(pl.col("dp_eq_bomb") == True).height,
      "| bomb_side == winner:", M.filter(pl.col("w_eq_bomb") == True).height)
print("  with elimination side:", M.filter(pl.col("elim_side").is_not_null()).height,
      "| elim == dp:", M.filter(pl.col("dp_eq_elim") == True).height,
      "| elim == winner:", M.filter(pl.col("w_eq_elim") == True).height)
print("  reason_side == winner_team_code in mismatch rows:", M.filter(pl.col("reason_eq_winner") == True).height,
      "of", M.height)
print("  mismatch rows by (winner, reason code, message, bomb_side):")
print(M.group_by(["winner_team_code", "win_reason_code", "win_reason_message", "bomb_side"]).len()
      .sort("len", descending=True).head(12))
print("  mismatch rows by cls:", M.group_by("cls").len().sort("cls").rows())

print("\n== CHECK 7: reason_side vs winner_team_code over ALL 5v5 rounds, by build ==")
print(Q.filter(pl.col("reason_side").is_not_null()).group_by(["build", "reason_eq_winner"]).len()
      .sort(["build", "reason_eq_winner"]).rows())

print("\n== CHECK 8: bomb event time-window test (is the explosion/defuse inside this round's tick window?) ==")
W = Q.filter(pl.col("bomb_side").is_not_null() & pl.col("exp_in_window").is_not_null())
W = W.with_columns(
    pl.when(pl.col("w_eq_dp") == False).then(pl.lit("win_ne_dp")).otherwise(pl.lit("win_eq_dp")).alias("grp"))
print(W.group_by(["build", "grp"]).agg(
    pl.len().alias("n"),
    pl.col("exp_in_window").cast(pl.Int64).sum().alias("exp_in_win"),
    pl.col("def_in_window").cast(pl.Int64).sum().alias("def_in_win"),
    pl.col("plant_in_window").cast(pl.Int64).sum().alias("plant_in_win"),
).sort(["build", "grp"]))
ob = Q.filter(pl.col("bomb_side").is_not_null() & pl.col("re_tick").is_not_null() & pl.col("exp_tick").is_not_null())
print("exploded rounds, round_end tick minus explosion tick (ticks; median, min, max) by build:",
      ob.filter(pl.col("bomb_side") == 2).with_columns((pl.col("re_tick") - pl.col("exp_tick")).alias("d"))
      .group_by("build").agg(pl.col("d").median().alias("median"), pl.col("d").min().alias("min"),
                             pl.col("d").max().alias("max"), pl.len().alias("n")).sort("build").rows())
print("defused rounds, round_end tick minus defuse tick (ticks; median, min, max) by build:",
      ob.filter(pl.col("bomb_side") == 3).with_columns((pl.col("re_tick") - pl.col("def_tick")).alias("d"))
      .group_by("build").agg(pl.col("d").median().alias("median"), pl.col("d").min().alias("min"),
                             pl.col("d").max().alias("max"), pl.len().alias("n")).sort("build").rows())

print("\n== CHECK 9: build vs date as the marker (round 13 and all fas), 5v5 dp known ==")
X = C1.filter(pl.col("cls") != "other")
print(X.group_by(["build", "after_onset_ts", "w_eq_dp"]).len().sort(["build", "after_onset_ts", "w_eq_dp"]).rows())
print("9842 matches after onset ts (5v5 matches):",
      Q.filter((pl.col("build") == "9842") & pl.col("after_onset_ts")).group_by("match_key").len().height)
print("9961 matches before onset ts (5v5 matches):",
      Q.filter((pl.col("build") == "9961") & (~pl.col("after_onset_ts"))).group_by("match_key").len().height)

print("\n== CHECK 10: halftime continuity of raw score fields (pair 12->13 and 24->25 and 27->28, baseline 5->6) ==")
PRS = Q.filter(pl.col("round").is_in([5, 12, 24, 27])).filter(pl.col("noswap_valid").is_not_null())
print(PRS.with_columns(
    pl.when(pl.col("noswap_valid") & pl.col("swap_valid")).then(pl.lit("both"))
      .when(pl.col("noswap_valid")).then(pl.lit("noswap_only"))
      .when(pl.col("swap_valid")).then(pl.lit("swap_only"))
      .otherwise(pl.lit("neither")).alias("pair_type"))
    .group_by(["build", "round", "pair_type"]).len().sort(["build", "round", "pair_type"]).rows())
print("schedule validity by pair round (5v5 pairs), build 9842 / 9961:")
SV = Q.filter(pl.col("sch_valid").is_not_null()).group_by(["build", "round"]).agg(
    pl.len().alias("n"), (pl.col("sch_valid") == False).sum().alias("invalid")).sort(["build", "round"])
print(SV.filter(pl.col("round").is_in([5, 12, 13, 24, 25, 27, 28, 30, 31])).rows())

print("\n== CHECK 11: t_score vs t_score_raw equality (5v5) by build ==")
T = Q.filter(pl.col("t_score").is_not_null())
print(T.group_by("build").agg(
    pl.len().alias("n"),
    (pl.col("t_score") == pl.col("t_score_raw")).sum().alias("t_eq_raw"),
    (pl.col("ct_score") == pl.col("ct_score_raw")).sum().alias("ct_eq_raw"),
).sort("build").rows())

print("\n== CHECK 12: non-fas, non-bomb, elimination-determined rounds where dp != winner (post build 9961) ==")
N = C1.filter((pl.col("build") == "9961") & (pl.col("cls") == "other") & pl.col("bomb_side").is_null())
print("non-fas non-bomb 9961 rounds with dp side:", N.height,
      "| winner != dp:", N.filter(pl.col("w_eq_dp") == False).height)
NE = N.filter(pl.col("elim_side").is_not_null())
print("  of which elimination-determined:", NE.height,
      "| winner == elim:", NE.filter(pl.col("w_eq_elim") == True).height,
      "| dp == elim:", NE.filter(pl.col("dp_eq_elim") == True).height)
NM = N.filter(pl.col("w_eq_dp") == False)
print("  winner!=dp non-fas non-bomb rows:", NM.height, "| with elimination side:",
      NM.filter(pl.col("elim_side").is_not_null()).height,
      "| of those winner==elim:", NM.filter(pl.col("w_eq_elim") == True).height,
      "| dp==elim:", NM.filter(pl.col("dp_eq_elim") == True).height)

print("\n== CHECK 13: null death team codes (elimination undercount risk) by build ==")
PDn = (pl.scan_parquet(sorted(glob.glob(str(BASE / f"basic_player_death{SUF}" / "dataframe_*"))))
       .select(["match_key", "round", "player_team_code", "player_id"]).collect())
HB = pl.scan_parquet(sorted(glob.glob(str(BASE / "header" / "dataframe_*")))).select(["key", "build_num"]).collect()
PDn = PDn.join(HB.rename({"key": "match_key"}), on="match_key", how="left").with_columns(
    pl.col("build_num").cast(pl.Utf8).alias("build"))
print(PDn.group_by("build").agg(pl.len().alias("deaths"),
                                pl.col("player_team_code").is_null().sum().alias("null_team")).sort("build").rows())
