r"""Verifier for D4-5, step D: independent ground truth from player deaths, all rounds, 2024.

For each (match, round) with a freeze_end and a round_end:
  live window = first round_freeze_end tick .. first round_end tick
  last_victim_side = player_team_code of the death with the largest tick inside the live window
  (2 = T, 3 = CT; deaths with other codes are ignored)
Under elimination logic the team whose player died last has lost the round, so the death-implied winner
is the opposite side: death_winner = 5 - last_victim_side.

Compares, per round number and day group:
  (a) round_end.winner_team_code vs death_winner
  (b) score side (freeze_end r -> r+1, unambiguous, swap rounds excluded with mr=24 as in D4) vs death_winner
  (c) win_reason_code cross-tab against round_end winner and against death_winner
Also a control: the same comparisons for the 2026 tome are NOT run here (see step E).

Reads projected columns only. Stdout only.
Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\verify_D4_5_d_deaths.py
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


def read_dir(fd, cols):
    fs = sorted(glob.glob(str(fd / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


def read(ch, cols):
    return read_dir(sorted(BASE.glob(f"basic_{ch}.*"))[0], cols)


def swap_expected(r, mr=24):
    return r == mr // 2 or (r > mr and (r - mr - 3) % 6 == 0)


def grp(day_expr):
    return (pl.when(day_expr == "12").then(pl.lit("d12"))
            .when(day_expr.is_in(["01", "02", "03", "04", "05", "06", "07"])).then(pl.lit("d01-07"))
            .otherwise(pl.lit("d13-25")))


# round_end: first row per (match, round) by tick
re_ = (read("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code"])
       .sort(["match_key", "round", "tick"]).unique(["match_key", "round"], keep="first")
       .rename({"tick": "end_tick"}))
re_ = re_.with_columns(pl.col("match_key").str.extract(r"csds/\d{4}/\d\d/(\d\d)/", 1).alias("day")).with_columns(grp(pl.col("day")).alias("grp"))

# freeze_end: first row per (match, round) by tick
fe = (read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
      .filter(pl.col("event_type") == "round_freeze_end").sort(["match_key", "round", "tick"])
      .unique(["match_key", "round"], keep="first").rename({"tick": "live_tick"}))

# bomb outcome per (match, round)
bs = read("bomb_state", ["match_key", "round", "tick", "event_type"]).filter(pl.col("event_type").is_in(OUTCOMES))
bo = bs.group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("bevs")).with_columns(
    pl.when(pl.col("bevs").list.contains("bomb_defused")).then(pl.lit(CT)).otherwise(pl.lit(T)).alias("bside"))

# deaths with side codes
de = read("player_death", ["match_key", "round", "tick", "player_team_code"]).filter(
    pl.col("player_team_code").is_in([T, CT]))
print(f"player_death rows (side-coded): {de.height}")

# death inside the live window of its round (use round column, check window)
W = fe.select(["match_key", "round", "live_tick"]).join(re_.select(["match_key", "round", "end_tick"]),
                                                       on=["match_key", "round"], how="inner")
DW = de.join(W, on=["match_key", "round"], how="inner").filter(
    (pl.col("tick") >= pl.col("live_tick")) & (pl.col("tick") <= pl.col("end_tick")))
lastd = DW.sort(["match_key", "round", "tick"]).group_by(["match_key", "round"]).agg(
    pl.col("player_team_code").last().alias("last_victim"),
    pl.col("tick").max().alias("last_death_tick"),
    pl.len().alias("n_deaths_in_window"),
    (pl.col("player_team_code") == T).sum().alias("t_deaths"),
    (pl.col("player_team_code") == CT).sum().alias("ct_deaths"),
)
print(f"rounds with >=1 in-window death: {lastd.height}")

# score side from freeze_end r -> r+1 (A13 method), first freeze_end rows
nxt = fe.select(["match_key", "round", "t_score", "ct_score"]).with_columns(
    (pl.col("round") - 1).alias("round")).rename({"t_score": "t2", "ct_score": "c2"})
SD = fe.select(["match_key", "round", "t_score", "ct_score"]).join(nxt, on=["match_key", "round"], how="inner").with_columns(
    (pl.col("t2") - pl.col("t_score")).alias("dT"), (pl.col("c2") - pl.col("ct_score")).alias("dC")).with_columns(
    pl.when((pl.col("dC") == 1) & (pl.col("dT") == 0)).then(pl.lit(CT))
    .when((pl.col("dT") == 1) & (pl.col("dC") == 0)).then(pl.lit(T)).otherwise(pl.lit(0)).alias("score_side"))
SD = SD.with_columns(pl.struct(["round"]).map_elements(lambda s: not swap_expected(s["round"]), return_dtype=pl.Boolean)
                     .alias("valid_swap"))

J = (re_.join(lastd, on=["match_key", "round"], how="left")
     .join(SD.select(["match_key", "round", "score_side", "valid_swap"]), on=["match_key", "round"], how="left")
     .join(bo.select(["match_key", "round", "bside"]), on=["match_key", "round"], how="left"))
J = J.with_columns((5 - pl.col("last_victim")).alias("death_winner"))

print("\n==== (1) win_reason_code vs round_end winner, all 2024 rounds (consistency of round_end labels) ====")
print(J.group_by(["win_reason_code", "winner_team_code"]).len().sort(["win_reason_code", "winner_team_code"]))

print("\n==== (2) bomb side vs round_end winner, all rounds with bomb outcome ====")
bb = J.filter(pl.col("bside").is_not_null())
print(bb.group_by(["grp", "bside", "winner_team_code"]).len().sort(["grp", "bside", "winner_team_code"]))

# elimination-ended rounds: no bomb outcome, deaths present
E = J.filter(pl.col("bside").is_null() & pl.col("last_victim").is_not_null())
print(f"\nrounds without bomb outcome but with in-window deaths: {E.height}")
print("\n==== (3) death-implied winner vs round_end winner, no-bomb rounds, by group ====")
print(E.group_by("grp").agg(
    pl.len().alias("n"),
    (pl.col("death_winner") == pl.col("winner_team_code")).mean().alias("death_eq_round_end"),
).sort("grp"))
print("\n==== (3b) same, by win_reason_code ====")
print(E.group_by(["grp", "win_reason_code"]).agg(
    pl.len().alias("n"),
    (pl.col("death_winner") == pl.col("winner_team_code")).mean().alias("death_eq_round_end"),
).sort(["grp", "win_reason_code"]))

# round-by-round for d13-25 and d12, and d01-07 (no-bomb rounds; death logic vs round_end vs score side)
R = E.filter(pl.col("score_side").is_not_null() & (pl.col("score_side") != 0) & pl.col("valid_swap"))
print("\n==== (4) per round number: no-bomb, valid (no swap), unambiguous score side; death-implied vs round_end vs score ====")
tab = R.group_by(["grp", "round"]).agg(
    pl.len().alias("n"),
    (pl.col("death_winner") != pl.col("winner_team_code")).sum().alias("death_ne_end"),
    (pl.col("score_side") != pl.col("death_winner")).sum().alias("score_ne_death"),
    (pl.col("score_side") != pl.col("winner_team_code")).sum().alias("score_ne_end"),
).sort(["grp", "round"])
print(tab.filter(pl.col("grp") != "d13-25").filter(pl.col("round") <= 30))
print(tab.filter(pl.col("grp") == "d13-25").filter(pl.col("round") <= 30))

# bomb rounds: score side vs bomb side vs round_end (valid)
B = J.filter(pl.col("bside").is_not_null() & pl.col("valid_swap") & pl.col("score_side").is_not_null() & (pl.col("score_side") != 0))
print("\n==== (5) bomb rounds (valid): score side vs bomb side vs round_end, round 13 and others ====")
print(B.with_columns(pl.when(pl.col("round") == 13).then(pl.lit("r13")).otherwise(pl.lit("other")).alias("rb")).group_by(["grp", "rb"]).agg(
    pl.len().alias("n"),
    (pl.col("score_side") != pl.col("bside")).sum().alias("score_ne_bomb"),
    (pl.col("winner_team_code") != pl.col("bside")).sum().alias("end_ne_bomb"),
).sort(["grp", "rb"]))

# save J summary for later steps (small)
SCRATCH = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
SCRATCH.mkdir(parents=True, exist_ok=True)
J.select(["match_key", "round", "grp", "winner_team_code", "win_reason_code", "end_tick", "last_victim",
          "last_death_tick", "death_winner", "score_side", "valid_swap", "bside"]).write_parquet(
    SCRATCH / "verify_D4_5_rounds_2024.parquet")
print(f"\nwrote scratch summary to {SCRATCH} (not in project)")
