r"""Verifier for D4-5, step H: what kind of rounds carry the score-vs-round_end disagreements that have NO
independent evidence (no bomb outcome row, no unique wipe)? 2024 days 12-25 (and day 01-07 control).

For each valid (no swap after r under mr=24), unambiguous score side round r in {9, 13, 25, 28, other}:
  class = bomb (bomb outcome row), wipe (no-bomb, unique 5-death side), none (neither)
Within class 'none' and the disputed subset (score != round_end), tabulates:
  round_end win_reason_code and winner, score side, number of in-window deaths, distinct victims per side,
  round length (round_end tick - freeze_end tick, in seconds at 64 tick).
Reads projected columns only. Stdout only.
Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\verify_D4_5_h_unexplained.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_cols(18)
pl.Config.set_tbl_width_chars(240)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
OUTCOMES = ["bomb_defused", "bomb_exploded"]
T, CT = 2, 3
TEAM_SIZE = 5  # ASSUMPTION


def read_dir(fd, cols):
    fs = sorted(glob.glob(str(fd / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


def read(ch, cols):
    return read_dir(sorted(BASE.glob(f"basic_{ch}.*"))[0], cols)


def swap_expected(r, mr=24):
    return r == mr // 2 or (r > mr and (r - mr - 3) % 6 == 0)


fe = (read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
      .filter(pl.col("event_type") == "round_freeze_end").sort(["match_key", "round", "tick"])
      .unique(["match_key", "round"], keep="first").rename({"tick": "live"}))
re_ = (read("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code"])
       .sort(["match_key", "round", "tick"]).unique(["match_key", "round"], keep="first").rename({"tick": "end"}))
bs = read("bomb_state", ["match_key", "round", "event_type"]).filter(pl.col("event_type").is_in(OUTCOMES))
bo = bs.group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("bevs")).with_columns(
    pl.when(pl.col("bevs").list.contains("bomb_defused")).then(pl.lit(CT)).otherwise(pl.lit(T)).alias("bside"))
de = read("player_death", ["match_key", "round", "tick", "player_id", "player_team_code"]).filter(
    pl.col("player_team_code").is_in([T, CT]))

W = fe.join(re_, on=["match_key", "round"], how="inner")
W = W.with_columns(pl.col("match_key").str.extract(r"csds/\d{4}/\d\d/(\d\d)/", 1).alias("day")).with_columns(
    pl.when(pl.col("day") == "12").then(pl.lit("d12"))
    .when(pl.col("day").is_in(["01", "02", "03", "04", "05", "06", "07"])).then(pl.lit("d01-07"))
    .otherwise(pl.lit("d13-25")).alias("grp"))
nxt = fe.select(["match_key", "round", "t_score", "ct_score"]).with_columns((pl.col("round") - 1).alias("round")).rename(
    {"t_score": "t2", "ct_score": "c2"})
sd = fe.select(["match_key", "round", "t_score", "ct_score"]).join(nxt, on=["match_key", "round"], how="inner").with_columns(
    (pl.col("t2") - pl.col("t_score")).alias("dT"), (pl.col("c2") - pl.col("ct_score")).alias("dC")).with_columns(
    pl.when((pl.col("dC") == 1) & (pl.col("dT") == 0)).then(pl.lit(CT))
    .when((pl.col("dT") == 1) & (pl.col("dC") == 0)).then(pl.lit(T)).otherwise(pl.lit(0)).alias("score_side"))
sd = sd.with_columns(pl.struct(["round"]).map_elements(lambda s: not swap_expected(s["round"]), return_dtype=pl.Boolean)
                     .alias("valid"))
W = W.join(sd.select(["match_key", "round", "score_side", "valid"]), on=["match_key", "round"], how="left")
W = W.join(bo.select(["match_key", "round", "bside"]), on=["match_key", "round"], how="left")
DW = de.join(W.select(["match_key", "round", "live", "end"]), on=["match_key", "round"], how="inner").filter(
    (pl.col("tick") >= pl.col("live")) & (pl.col("tick") <= pl.col("end")))
dd = DW.group_by(["match_key", "round"]).agg(pl.col("player_id").n_unique().alias("n_victims"),
                                              (pl.col("player_team_code") == T).sum().alias("t_deaths"),
                                              (pl.col("player_team_code") == CT).sum().alias("ct_deaths"))
vic = DW.group_by(["match_key", "round", "player_team_code"]).agg(pl.col("player_id").n_unique().alias("nv")).pivot(
    on="player_team_code", index=["match_key", "round"], values="nv")
vic = vic.rename({"2": "nT", "3": "nCT"}) if "2" in vic.columns else vic
W = W.join(vic, on=["match_key", "round"], how="left").with_columns(
    pl.col("nT").fill_null(0) if "nT" in vic.columns else pl.lit(0).alias("nT"),
    pl.col("nCT").fill_null(0) if "nCT" in vic.columns else pl.lit(0).alias("nCT"))
W = W.with_columns(
    pl.when((pl.col("nT") >= TEAM_SIZE) & (pl.col("nCT") < TEAM_SIZE)).then(pl.lit(CT))
    .when((pl.col("nCT") >= TEAM_SIZE) & (pl.col("nT") < TEAM_SIZE)).then(pl.lit(T)).otherwise(pl.lit(0)).alias("wipe_winner"))
W = W.with_columns(
    pl.when(pl.col("round") == 13).then(pl.lit("r13"))
    .when(pl.col("round").is_in([9, 25, 28])).then(pl.lit("r9_25_28"))
    .otherwise(pl.lit("other")).alias("rclass"),
    pl.when(pl.col("bside").is_not_null()).then(pl.lit("bomb"))
    .when(pl.col("wipe_winner") != 0).then(pl.lit("wipe"))
    .otherwise(pl.lit("none")).alias("evid"),
    ((pl.col("end") - pl.col("live")) / 64.0).alias("secs"))
W = W.join(dd, on=["match_key", "round"], how="left")

V = W.filter(pl.col("valid") & (pl.col("score_side") != 0))
V = V.with_columns((pl.col("score_side") != pl.col("winner_team_code")).alias("dis"))
print("valid unambiguous rounds by evidence class and round class, 2024 d12-25 and d01-07:")
print(V.group_by(["grp", "rclass", "evid"]).agg(pl.len().alias("n"), pl.col("dis").sum().alias("disagree"),
                                                 pl.col("dis").mean().alias("rate")).sort(["grp", "rclass", "evid"]))

N = V.filter((pl.col("evid") == "none") & (pl.col("grp") != "d01-07"))
print("\nclass 'none' (no bomb, no unique wipe), d12-25, by round class and disagreement:")
print(N.group_by(["rclass", "dis"]).agg(pl.len().alias("n")).sort(["rclass", "dis"]))

print("\nclass 'none' disputed rounds (score != round_end): round_end win_reason_code x winner x score side, round 9/25/28:")
print(N.filter((pl.col("rclass") == "r9_25_28") & pl.col("dis")).group_by(["round", "win_reason_code", "winner_team_code", "score_side"]).len().sort(["round", "win_reason_code", "winner_team_code", "score_side"]))
print("\nclass 'none' disputed rounds: round 13:")
print(N.filter((pl.col("rclass") == "r13") & pl.col("dis")).group_by(["win_reason_code", "winner_team_code", "score_side"]).len().sort(["win_reason_code", "winner_team_code", "score_side"]))

print("\nclass 'none' disputed rounds: deaths and round length (secs), vs agreeing class 'none' rounds, by rclass:")
print(N.group_by(["rclass", "dis"]).agg(
    pl.len().alias("n"), pl.col("n_victims").median().alias("med_victims"),
    pl.col("t_deaths").median().alias("med_t_deaths"), pl.col("ct_deaths").median().alias("med_ct_deaths"),
    pl.col("secs").median().alias("med_secs"), pl.col("secs").quantile(0.9).alias("p90_secs"),
    (pl.col("n_victims").is_null()).sum().alias("no_deaths_rows")).sort(["rclass", "dis"]))

print("\nreason-code composition of class 'none' rounds, round 9 vs others (all valid, d12-25):")
print(N.with_columns(pl.when(pl.col("round") == 9).then(pl.lit("r9")).otherwise(pl.lit("other")).alias("r9")).group_by(
    ["r9", "win_reason_code", "dis"]).len().sort(["r9", "win_reason_code", "dis"]))

# 2024 round 9 disputed rounds: does the score say the side that round_end says lost on time-out?
D9 = N.filter((pl.col("round") == 9) & pl.col("dis"))
print(f"\nround 9 disputed, class none, d12-25: n={D9.height}")
print(D9.group_by(["win_reason_code", "winner_team_code", "score_side"]).len().sort(["win_reason_code", "winner_team_code", "score_side"]))
print("secs quantiles (disputed r9 none):")
print(D9.select([pl.col("secs").quantile(q).alias(f"q{int(q*100)}") for q in [0.1, 0.5, 0.9]]))

# save a small scratch table for the record
SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
SCR.mkdir(parents=True, exist_ok=True)
V.select(["match_key", "round", "grp", "rclass", "evid", "winner_team_code", "score_side", "win_reason_code",
          "bside", "wipe_winner", "nT", "nCT", "secs", "dis"]).write_parquet(SCR / "verify_D4_5_valid_rounds_2024.parquet")
print(f"\nscratch: {SCR}")
