r"""D4 check: for the 2024 rounds flagged by the score test (9, 13, 25, 28; days 12-25), does the last death in the
live window (any gap, A7) agree with the score side or with round_end? Also reports the same for unflagged rounds
(control: 1, 2, 10, 14, 19) to show the death record tracks the score side wherever round_end agrees with it.
Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\D4_flagged_rounds_death.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(12)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
T, CT = 2, 3
ROUNDS = [1, 2, 9, 10, 13, 14, 19, 25, 28]


def read_dir(fd, cols):
    fs = sorted(glob.glob(str(fd / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


def read(ch, cols):
    return read_dir(sorted(BASE.glob(f"basic_{ch}.*"))[0], cols)


rs = read("round_state", ["match_key", "round", "event_type", "t_score", "ct_score"])
fe = (rs.filter(pl.col("event_type") == "round_freeze_end").select(["match_key", "round", "t_score", "ct_score"])
      .unique(["match_key", "round"], keep="first"))
nx = fe.with_columns((pl.col("round") - 1).alias("round")).rename({"t_score": "t2", "ct_score": "c2"})
D = fe.join(nx, on=["match_key", "round"], how="inner").with_columns(
    (pl.col("t2") - pl.col("t_score")).alias("dT"), (pl.col("c2") - pl.col("ct_score")).alias("dC")).with_columns(
    pl.when((pl.col("dC") == 1) & (pl.col("dT") == 0)).then(pl.lit(CT))
    .when((pl.col("dT") == 1) & (pl.col("dC") == 0)).then(pl.lit(T)).otherwise(pl.lit(0)).alias("score_side"))
D = D.filter(pl.col("round").is_in(ROUNDS) & (pl.col("score_side") != 0)).with_columns(
    pl.col("match_key").str.extract(r"csds/\d{4}/\d\d/(\d\d)/", 1).cast(pl.Int32).alias("dd")).filter(pl.col("dd") >= 12)
live = (rs.filter(pl.col("event_type") == "round_freeze_end").select(["match_key", "round", pl.col("t_score") * 0 + 0])
        .unique(["match_key", "round"], keep="first"))
lv = read("round_state", ["match_key", "round", "tick", "event_type"]).filter(
    pl.col("event_type") == "round_freeze_end").select(["match_key", "round", pl.col("tick").alias("live")]).unique(
    ["match_key", "round"], keep="first")
re_ = read("round_end", ["match_key", "round", "tick", "winner_team_code", "player_count"]).unique(
    ["match_key", "round"], keep="first").rename({"tick": "end"})
de = read("player_death", ["match_key", "round", "tick", "player_team_code"]).filter(
    pl.col("player_team_code").is_in([T, CT]))
J = D.join(lv, on=["match_key", "round"], how="inner").join(re_, on=["match_key", "round"], how="inner")
J = J.filter(pl.col("player_count") == 20)
DD = de.join(J.select(["match_key", "round", "live", "end"]), on=["match_key", "round"], how="inner").filter(
    (pl.col("tick") >= pl.col("live")) & (pl.col("tick") <= pl.col("end")))
lt = DD.group_by(["match_key", "round"]).agg(pl.col("tick").max().alias("lt"))
lv2 = DD.join(lt, on=["match_key", "round"]).filter(pl.col("tick") == pl.col("lt")).group_by(["match_key", "round"]).agg(
    pl.col("player_team_code").n_unique().alias("ns"), pl.col("player_team_code").first().alias("victim"), pl.col("lt").first())
K = J.join(lv2.filter(pl.col("ns") == 1), on=["match_key", "round"], how="inner").with_columns(
    (pl.col("end") - pl.col("lt")).alias("gap"),
    (5 - pl.col("victim")).alias("pred_victim"),
    (pl.col("score_side") != pl.col("winner_team_code")).alias("flag"))
K = K.with_columns((pl.col("pred_victim") == pl.col("score_side")).alias("victim_eq_score"),
                   (pl.col("pred_victim") == pl.col("winner_team_code")).alias("victim_eq_round_end"))
print("2024 days 12-25, rounds with a last death in window: victim-side prediction vs score side and vs round_end")
print(K.group_by(["round", "flag"]).agg(
    pl.len().alias("n"),
    pl.col("victim_eq_score").mean().alias("victim_eq_score"),
    pl.col("victim_eq_round_end").mean().alias("victim_eq_round_end"),
    pl.col("gap").median().alias("median_gap_ticks")).sort(["round", "flag"]))
