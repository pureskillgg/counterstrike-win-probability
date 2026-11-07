r"""Verifier for D4-5, step C: per-match round timelines for disputed 2024 round-13 matches.

Shows, for a few matches where the D4 score-side (freeze_end 13 -> 14) disagrees with round_end winner at round 13:
  freeze_end rows per round (count, ticks), t_score/ct_score at each freeze_end, round_end tick and winner,
  bomb outcome per round, and the header starters' final-score sum.
Also counts freeze_end duplicates across all 2024 matches (per match-round), by day group.
Reads only projected columns. Output is stdout only.
Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\verify_D4_5_c_timeline.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(120)
pl.Config.set_tbl_cols(16)
pl.Config.set_fmt_str_lengths(40)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
OUTCOMES = ["bomb_defused", "bomb_exploded"]
T, CT = 2, 3


def read_dir(fd, cols):
    fs = sorted(glob.glob(str(fd / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


def read(ch, cols):
    return read_dir(sorted(BASE.glob(f"basic_{ch}.*"))[0], cols)


H = read_dir(BASE / "header", ["key", "t_starters_score_final", "ct_starters_score_final"]).with_columns(
    pl.col("key").str.extract(r"csds/\d{4}/\d\d/(\d\d)/", 1).alias("day"),
    (pl.col("t_starters_score_final") + pl.col("ct_starters_score_final")).alias("fsum"))

fe_all = read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"]).filter(
    pl.col("event_type") == "round_freeze_end")
print(f"2024 round_freeze_end rows: {fe_all.height}")
fe_cnt = fe_all.group_by(["match_key", "round"]).agg(pl.len().alias("n_fe"))
print("freeze_end rows per (match, round):")
print(fe_cnt.group_by("n_fe").len().sort("n_fe"))
dup_keys = fe_cnt.filter(pl.col("n_fe") > 1)
print(f"match-rounds with >1 freeze_end: {dup_keys.height}")
print(dup_keys.group_by("round").len().sort("round").head(40))

# keep every freeze_end row (not just first) for the timeline view
fe_ex = (fe_all.join(dup_keys.select(["match_key", "round"]), on=["match_key", "round"], how="inner"))
print(f"\nrows in dup match-rounds: {fe_ex.height}")
print(fe_ex.join(H.select(["key", "day"]), left_on="match_key", right_on="key", how="left")
      .group_by("day").len().sort("day"))

# disputed matches: round 13 score side vs round_end, using first freeze_end (as D4 does)
fe1 = fe_all.unique(["match_key", "round"], keep="first").select(["match_key", "round", "t_score", "ct_score"])
nxt = fe1.with_columns((pl.col("round") - 1).alias("round")).rename({"t_score": "t2", "ct_score": "c2"})
D = fe1.join(nxt, on=["match_key", "round"], how="inner").with_columns(
    (pl.col("t2") - pl.col("t_score")).alias("dT"), (pl.col("c2") - pl.col("ct_score")).alias("dC")).with_columns(
    pl.when((pl.col("dC") == 1) & (pl.col("dT") == 0)).then(pl.lit(CT))
    .when((pl.col("dT") == 1) & (pl.col("dC") == 0)).then(pl.lit(T)).otherwise(pl.lit(0)).alias("score_side"))
re_ = read("round_end", ["match_key", "round", "tick", "winner_team_code"]).unique(["match_key", "round"], keep="first")
J13 = D.filter((pl.col("round") == 13) & (pl.col("score_side") != 0)).join(
    re_.filter(pl.col("round") == 13), on=["match_key", "round"], how="inner").filter(
    pl.col("score_side") != pl.col("winner_team_code"))
J13 = J13.join(H.select(["key", "day", "fsum"]), left_on="match_key", right_on="key", how="left")
dis = J13.filter(pl.col("day") == "12").head(4)
print(f"\ndisputed round-13 matches (day 12 sample of {dis.height}):")
print(dis.select(["match_key", "round", "score_side", "winner_team_code", "fsum"]))
dis_keys = J13.filter(pl.col("day").is_in(["12", "13", "14"])).head(3).select("match_key").to_series().to_list()
dis_keys += J13.filter(pl.col("day") == "17").head(2).select("match_key").to_series().to_list()
print("\nsample disputed match keys:")
for k in dis_keys:
    print("  ", k)

bs = read("bomb_state", ["match_key", "round", "tick", "event_type"]).filter(pl.col("event_type").is_in(OUTCOMES))
endt = re_.select(["match_key", "round", pl.col("tick").alias("end_tick"), "winner_team_code"])
for k in dis_keys:
    print(f"\n=================== {k} ===================")
    h = H.filter(pl.col("key") == k).select(["fsum", "t_starters_score_final", "ct_starters_score_final"])
    print("header:", h.row(0) if h.height else None)
    rr = fe_all.filter(pl.col("match_key") == k).group_by("round").agg(
        pl.len().alias("n_fe"), pl.col("tick").min().alias("fe_tick_min"), pl.col("tick").max().alias("fe_tick_max"),
        pl.col("t_score").first().alias("t_sc"), pl.col("ct_score").first().alias("ct_sc"))
    rr = rr.join(endt.filter(pl.col("match_key") == k).select(["round", "end_tick", "winner_team_code"]), on="round", how="full",
                 coalesce=True).sort("round")
    bo = bs.filter(pl.col("match_key") == k).group_by("round").agg(pl.col("event_type").unique().alias("bomb_evs"))
    rr = rr.join(bo, on="round", how="left").sort("round")
    print(rr)
