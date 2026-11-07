r"""Verifier for D4-5, step E: two independent tests.

E1 (wipe test, no score fields and no round_end winner used to define the truth):
  In a no-bomb round, the side with >= 5 distinct dead players inside the live window (freeze_end .. round_end)
  is treated as eliminated (ASSUMPTION: 5-player teams, MR12 CS2 rosters; sides with <5 distinct victims are
  "not wiped"). Winner-by-wipe = opposite side of the wiped side, only when exactly one side is wiped.
  Compared with round_end.winner_team_code and with the score side (freeze_end r -> r+1, unambiguous).

E2 (alignment test): for bomb_state rows (outcome events and planted) and player_death rows labelled round r,
  is the tick inside the window of round r (freeze_end r .. round_end r), of a neighbouring round, or outside all?
  A row-round misalignment in 2024 days 12-25 would show as many rows outside their own window.

Reads projected columns only. Stdout only (plus a small scratch file in the session scratchpad).
Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\verify_D4_5_e_wipe_align.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(120)
pl.Config.set_tbl_cols(16)
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


def grp_of(day):
    return (pl.when(day == "12").then(pl.lit("d12"))
            .when(day.is_in(["01", "02", "03", "04", "05", "06", "07"])).then(pl.lit("d01-07"))
            .otherwise(pl.lit("d13-25")))


def swap_expected(r, mr=24):
    return r == mr // 2 or (r > mr and (r - mr - 3) % 6 == 0)


fe = (read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
      .filter(pl.col("event_type") == "round_freeze_end").sort(["match_key", "round", "tick"])
      .unique(["match_key", "round"], keep="first").rename({"tick": "live"}))
re_ = (read("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code"])
       .sort(["match_key", "round", "tick"]).unique(["match_key", "round"], keep="first").rename({"tick": "end"}))
win = fe.select(["match_key", "round", "live", "t_score", "ct_score"]).join(
    re_, on=["match_key", "round"], how="inner")
win = win.with_columns(pl.col("match_key").str.extract(r"csds/\d{4}/\d\d/(\d\d)/", 1).alias("day")).with_columns(
    grp_of(pl.col("day")).alias("grp"))

# score side
nxt = fe.select(["match_key", "round", "t_score", "ct_score"]).with_columns((pl.col("round") - 1).alias("round")).rename(
    {"t_score": "t2", "ct_score": "c2"})
sd = fe.select(["match_key", "round", "t_score", "ct_score"]).join(nxt, on=["match_key", "round"], how="inner").with_columns(
    (pl.col("t2") - pl.col("t_score")).alias("dT"), (pl.col("c2") - pl.col("ct_score")).alias("dC")).with_columns(
    pl.when((pl.col("dC") == 1) & (pl.col("dT") == 0)).then(pl.lit(CT))
    .when((pl.col("dT") == 1) & (pl.col("dC") == 0)).then(pl.lit(T)).otherwise(pl.lit(0)).alias("score_side"))
sd = sd.with_columns(pl.struct(["round"]).map_elements(lambda s: not swap_expected(s["round"]), return_dtype=pl.Boolean)
                     .alias("valid"))
win = win.join(sd.select(["match_key", "round", "score_side", "valid"]), on=["match_key", "round"], how="left")

bs = read("bomb_state", ["match_key", "round", "tick", "event_type"])
bo = (bs.filter(pl.col("event_type").is_in(OUTCOMES)).group_by(["match_key", "round"])
      .agg(pl.col("event_type").unique().alias("bevs")).with_columns(
    pl.when(pl.col("bevs").list.contains("bomb_defused")).then(pl.lit(CT)).otherwise(pl.lit(T)).alias("bside")))
win = win.join(bo.select(["match_key", "round", "bside"]), on=["match_key", "round"], how="left")

de = read("player_death", ["match_key", "round", "tick", "player_id", "player_team_code"]).filter(
    pl.col("player_team_code").is_in([T, CT]))

# E1: distinct victims per side inside the live window (using the round's own window)
DW = de.join(win.select(["match_key", "round", "live", "end"]), on=["match_key", "round"], how="inner").filter(
    (pl.col("tick") >= pl.col("live")) & (pl.col("tick") <= pl.col("end")))
vic = DW.group_by(["match_key", "round", "player_team_code"]).agg(pl.col("player_id").n_unique().alias("nv")).pivot(
    on="player_team_code", index=["match_key", "round"], values="nv").rename({"2": "nT", "3": "nCT"}).fill_null(0)
win = win.join(vic, on=["match_key", "round"], how="left").with_columns(
    pl.col("nT").fill_null(0), pl.col("nCT").fill_null(0))
win = win.with_columns(
    (pl.col("nT") >= TEAM_SIZE).alias("T_wiped"), (pl.col("nCT") >= TEAM_SIZE).alias("CT_wiped")).with_columns(
    pl.when(pl.col("T_wiped") & ~pl.col("CT_wiped")).then(pl.lit(CT))
    .when(pl.col("CT_wiped") & ~pl.col("T_wiped")).then(pl.lit(T)).otherwise(pl.lit(0)).alias("wipe_winner"))
win = win.with_columns(pl.when(pl.col("round") == 13).then(pl.lit("r13"))
                       .when(pl.col("round").is_in([9, 25, 28])).then(pl.lit("r9_25_28"))
                       .otherwise(pl.lit("other")).alias("rclass"))

noba = win.filter(pl.col("bside").is_null() & pl.col("valid") & (pl.col("score_side") != 0) & (pl.col("wipe_winner") != 0))
print("==== E1 wipe test: no-bomb, valid, unambiguous score side, exactly one side wiped ====")
print("n rounds:", noba.height)
print(noba.group_by(["grp", "rclass"]).agg(
    pl.len().alias("n"),
    (pl.col("wipe_winner") == pl.col("winner_team_code")).mean().alias("wipe_eq_round_end"),
    (pl.col("wipe_winner") == pl.col("score_side")).mean().alias("wipe_eq_score"),
    (pl.col("score_side") == pl.col("winner_team_code")).mean().alias("score_eq_round_end"),
).sort(["grp", "rclass"]))

# disagreement breakdown for the special rounds: where score and round_end differ, which side does the wipe support?
dis = noba.filter(pl.col("score_side") != pl.col("winner_team_code"))
print("\nE1b: rounds where score side != round_end winner (no-bomb, valid), wipe-implied winner support:")
print(dis.group_by(["grp", "rclass"]).agg(
    pl.len().alias("n_disagree"),
    (pl.col("wipe_winner") == pl.col("score_side")).sum().alias("wipe_supports_score"),
    (pl.col("wipe_winner") == pl.col("winner_team_code")).sum().alias("wipe_supports_round_end"),
).sort(["grp", "rclass"]))

# coverage of wipe evidence in the special rounds
spec = win.filter(pl.col("grp") != "d01-07").filter(pl.col("rclass") != "other").filter(pl.col("bside").is_null())
print("\ncoverage: no-bomb special rounds (d12-25) with a unique wipe (any score side):",
      spec.filter(pl.col("wipe_winner") != 0).height, "of", spec.height)

# E2: alignment of bomb_state and player_death rows to their own round window
W2 = fe.select(["match_key", "round", "live"]).join(re_.select(["match_key", "round", "end"]), on=["match_key", "round"], how="inner")
W2 = W2.with_columns(pl.col("match_key").str.extract(r"csds/\d{4}/\d\d/(\d\d)/", 1).alias("day")).with_columns(
    grp_of(pl.col("day")).alias("grp"))


def align(rows, label):
    rows = rows.join(W2.select(["match_key", "round", "live", "end", "grp"]).rename({"round": "rnd"}).with_columns(pl.col("rnd")),
                     on="match_key", how="inner")
    # rows with label round r; window of round r
    own = rows.join(W2.select(["match_key", "round", "live", "end"]).rename({"round": "rnd", "live": "lv", "end": "en"}),
                    left_on=["match_key", "round"], right_on=["match_key", "rnd"], how="inner")
    own = own.with_columns(((pl.col("tick") >= pl.col("lv")) & (pl.col("tick") <= pl.col("en"))).alias("in_own"))
    return own


bsr = bs.select(["match_key", "round", "tick", "event_type"]).filter(pl.col("event_type").is_in(OUTCOMES + ["bomb_planted"]))
ob = (bsr.join(W2.select(["match_key", "round", "live", "end", "grp"]), on=["match_key", "round"], how="inner")
      .with_columns(((pl.col("tick") >= pl.col("live")) & (pl.col("tick") <= pl.col("end"))).alias("in_own")))
dr = de.select(["match_key", "round", "tick"]).join(W2.select(["match_key", "round", "live", "end", "grp"]),
                                                    on=["match_key", "round"], how="inner").with_columns(
    ((pl.col("tick") >= pl.col("live")) & (pl.col("tick") <= pl.col("end"))).alias("in_own"))
print("\n==== E2 alignment: fraction of rows labelled round r whose tick lies inside the window of round r ====")
print("bomb outcome/plant rows, by group and round class:")
ob = ob.with_columns(pl.when(pl.col("round") == 13).then(pl.lit("r13"))
                     .when(pl.col("round").is_in([9, 25, 28])).then(pl.lit("r9_25_28")).otherwise(pl.lit("other")).alias("rclass"))
print(ob.group_by(["grp", "rclass"]).agg(pl.len().alias("n"), pl.col("in_own").mean().alias("frac_in_own_window")).sort(["grp", "rclass"]))
dr = dr.with_columns(pl.when(pl.col("round") == 13).then(pl.lit("r13"))
                     .when(pl.col("round").is_in([9, 25, 28])).then(pl.lit("r9_25_28")).otherwise(pl.lit("other")).alias("rclass"))
print("player_death rows, by group and round class:")
print(dr.group_by(["grp", "rclass"]).agg(pl.len().alias("n"), pl.col("in_own").mean().alias("frac_in_own_window")).sort(["grp", "rclass"]))

# round_end tick monotonicity within match: does round r end before round r+1 freeze_end?
mono = (re_.select(["match_key", "round", "end"]).join(
    fe.select(["match_key", "round", "live"]).with_columns((pl.col("round") - 1).alias("round")).rename({"live": "live_next"}),
    on=["match_key", "round"], how="inner"))
print("\nround_end(r) tick < freeze_end(r+1) tick, fraction, by round class:")
mono = mono.with_columns(pl.when(pl.col("round") + 1 == 13).then(pl.lit("r13"))
                         .when((pl.col("round") + 1).is_in([9, 25, 28])).then(pl.lit("r9_25_28")).otherwise(pl.lit("other")).alias("rclass"))
mono = mono.with_columns(
    pl.col("match_key").str.extract(r"csds/\d{4}/\d\d/(\d\d)/", 1).alias("day")).with_columns(grp_of(pl.col("day")).alias("grp"))
print(mono.group_by(["grp", "rclass"]).agg(pl.len().alias("n"), (pl.col("end") < pl.col("live_next")).mean().alias("frac_end_before_next_fe")).sort(["grp", "rclass"]))
