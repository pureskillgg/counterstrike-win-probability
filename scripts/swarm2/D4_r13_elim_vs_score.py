r"""D4 check: on round-13 elimination-ended rounds, do the death records agree with the score-channel winner?

Round 13 is valid for the score delta (no swap after round 13). A round is elimination-ended if its last death
in the live window is within 192 ticks of round_end (A10). score_side from round_state freeze_end 13 -> 14 (A13).
Predicted winner from the last victim side = 5 - victim side (A7).
Compares, per era and day group: score_side vs round_end winner, and score_side vs death-predicted winner.
Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\D4_r13_elim_vs_score.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(14)
ROOT = Path(r"G:\pureskill\csds-cs2")
TOMES = {"2024": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
         "2026": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"}
T, CT = 2, 3
MAX_GAP = 192


def read_dir(fd, cols):
    fs = sorted(glob.glob(str(fd / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


def read(base, ch, cols):
    return read_dir(sorted(base.glob(f"basic_{ch}.*"))[0], cols)


for era, base in TOMES.items():
    print(f"\n==================== ERA {era}, round 13 ====================")
    rs = read(base, "round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
    fe = (rs.filter(pl.col("event_type") == "round_freeze_end")
          .select(["match_key", "round", "tick", "t_score", "ct_score"]).unique(["match_key", "round"], keep="first"))
    r13 = fe.filter(pl.col("round") == 13).rename({"tick": "live"})
    r14 = fe.filter(pl.col("round") == 14).select(["match_key", pl.col("t_score").alias("t14"), pl.col("ct_score").alias("ct14")])
    S = r13.join(r14, on="match_key", how="inner").with_columns(
        (pl.col("ct14") - pl.col("ct_score")).alias("dC"), (pl.col("t14") - pl.col("t_score")).alias("dT")).with_columns(
        pl.when((pl.col("dC") == 1) & (pl.col("dT") == 0)).then(pl.lit(CT))
        .when((pl.col("dT") == 1) & (pl.col("dC") == 0)).then(pl.lit(T)).otherwise(pl.lit(0)).alias("score_side"))
    re_ = read(base, "round_end", ["match_key", "round", "tick", "winner_team_code", "player_count"]).unique(
        ["match_key", "round"], keep="first").rename({"tick": "end"})
    de = read(base, "player_death", ["match_key", "round", "tick", "player_team_code"]).filter(
        pl.col("player_team_code").is_in([T, CT]))
    J = S.join(re_.filter(pl.col("round") == 13), on=["match_key", "round"], how="inner").filter(pl.col("score_side") != 0)
    D = de.filter(pl.col("round") == 13).join(J.select(["match_key", "live", "end"]), on="match_key", how="inner").filter(
        (pl.col("tick") >= pl.col("live")) & (pl.col("tick") <= pl.col("end")))
    last = D.group_by("match_key").agg(pl.col("tick").max().alias("last_tick"))
    lastD = D.join(last, on="match_key").filter(pl.col("tick") == pl.col("last_tick")).group_by("match_key").agg(
        pl.col("player_team_code").first().alias("victim"), pl.col("last_tick").first(), pl.len().alias("nlast"))
    K = J.join(lastD, on="match_key", how="inner").filter(pl.col("nlast") == 1).with_columns(
        (pl.col("end") - pl.col("last_tick")).alias("gap")).filter(pl.col("gap") <= MAX_GAP).with_columns(
        (5 - pl.col("victim")).alias("pred_death"),
        pl.col("match_key").str.extract(r"csds/\d{4}/\d\d/(\d\d)/", 1).alias("day"),
        pl.when(pl.col("match_key").str.extract(r"csds/\d{4}/\d\d/(\d\d)/", 1).cast(pl.Int32) <= 7)
        .then(pl.lit("d01-07")).otherwise(pl.lit("d12-25")).alias("dg"))
    K = K.with_columns((pl.col("score_side") == pl.col("winner_team_code")).alias("score_eq_end"),
                       (pl.col("score_side") == pl.col("pred_death")).alias("score_eq_death"))
    print(f"round-13 elimination-ended rounds with score side: {K.height}")
    print(K.group_by("dg").agg(pl.len().alias("n"), pl.col("score_eq_end").mean().alias("score_eq_round_end"),
                               pl.col("score_eq_death").mean().alias("score_eq_death_pred")).sort("dg"))
    print("by day (2024 only shows the defect days):")
    print(K.group_by("day").agg(pl.len().alias("n"), pl.col("score_eq_end").mean().alias("score_eq_round_end"),
                                pl.col("score_eq_death").mean().alias("score_eq_death_pred")).sort("day"))
