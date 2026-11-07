"""D3-2 verifier diagnostic: raw per-round sequences around round 13 for sample matches (2024 G: tome).

Read-only. Prints, for sample matches with a round-13 bomb outcome and for sample matches with a
round-13 round_end/score disagreement and no bomb, the round sequence:
  round, round_end winner (pc20), reason, bomb outcome, score-chain winner (cleaned), freeze t/ct score.
Also prints the round-to-round agreement of consecutive score-chain winners and of round_end winners,
to test whether the 'score(r-1) == bomb(r)' result of verify_D3_2_main section 5 is a real pattern.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(220)

TOME = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
T, CT = 2, 3


def load(ch, cols):
    fs = sorted(glob.glob(str(TOME / f"basic_{ch}{SUF}" / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


RE = load("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count"])
BS = load("bomb_state", ["match_key", "round", "tick", "event_type"])
RS = load("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
DAY = pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).cast(pl.Int64)

fe = (RS.filter(pl.col("event_type") == "round_freeze_end").sort("tick")
      .group_by(["match_key", "round"])
      .agg(pl.col("t_score").first().alias("t_b"), pl.col("ct_score").first().alias("c_b"),
           pl.col("tick").first().alias("fe_tick")))
nxt = fe.select(["match_key", (pl.col("round") - 1).alias("round"),
                 pl.col("t_b").alias("t_a"), pl.col("c_b").alias("c_a")])
sc = fe.join(nxt, on=["match_key", "round"], how="left").with_columns(
    (pl.col("t_a") - pl.col("t_b")).alias("dt"), (pl.col("c_a") - pl.col("c_b")).alias("dc")).with_columns(
    pl.when((pl.col("dc") == 1) & (pl.col("dt") == 0)).then(pl.lit(CT, dtype=pl.Int64))
    .when((pl.col("dt") == 1) & (pl.col("dc") == 0)).then(pl.lit(T, dtype=pl.Int64))
    .otherwise(pl.lit(None, dtype=pl.Int64)).alias("sc_win"))

re20 = (RE.filter(pl.col("player_count") == 20).sort("tick").group_by(["match_key", "round"])
        .agg(pl.col("winner_team_code").first().alias("re_w"), pl.col("win_reason_code").first().alias("reason")))
outc = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
        .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
outc = outc.with_columns(pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(CT))
                         .otherwise(pl.lit(T)).alias("bomb_side")).select(["match_key", "round", "bomb_side"])

ALL = (re20.join(sc.select(["match_key", "round", "t_b", "c_b", "sc_win"]), on=["match_key", "round"], how="left")
       .join(outc, on=["match_key", "round"], how="left")
       .with_columns(DAY.alias("day")))

# (a) consecutive-round agreement of the score chain and of round_end, days 12-25, rounds with both defined
Wsc = ALL.select(["match_key", "round", "sc_win", "re_w"])
pair = Wsc.join(Wsc.select(["match_key", (pl.col("round") + 1).alias("round"),
                            pl.col("sc_win").alias("sc_next"), pl.col("re_w").alias("re_next")]),
                on=["match_key", "round"], how="inner").join(
    ALL.select(["match_key", "round", "day"]), on=["match_key", "round"], how="left").filter(pl.col("day") >= 12)
pp = pair.filter(pl.col("sc_win").is_not_null() & pl.col("sc_next").is_not_null())
print("== (a) P(winner(r) == winner(r+1)) days 12-25, by r (score chain | round_end)")
print(pp.group_by("round").agg(
    pl.len().alias("n"),
    (pl.col("sc_win") == pl.col("sc_next")).mean().alias("score_same_as_next"),
    ((pl.col("re_w") == pl.col("re_next"))).mean().alias("re_same_as_next")).sort("round").head(16))

# (b) same for days 01-07 (control period where round 13 is clean)
pair7 = Wsc.join(Wsc.select(["match_key", (pl.col("round") + 1).alias("round"),
                             pl.col("sc_win").alias("sc_next"), pl.col("re_w").alias("re_next")]),
                 on=["match_key", "round"], how="inner").join(
    ALL.select(["match_key", "round", "day"]), on=["match_key", "round"], how="left").filter(pl.col("day") < 12)
pp7 = pair7.filter(pl.col("sc_win").is_not_null() & pl.col("sc_next").is_not_null())
print("\n== (b) same, days 01-07")
print(pp7.group_by("round").agg(
    pl.len().alias("n"),
    (pl.col("sc_win") == pl.col("sc_next")).mean().alias("score_same_as_next"),
    ((pl.col("re_w") == pl.col("re_next"))).mean().alias("re_same_as_next")).sort("round").head(16))

# (c) raw sequences for sample matches with a round-13 bomb outcome, days 12-25
r13b = ALL.filter((pl.col("round") == 13) & (pl.col("day") >= 12) & pl.col("bomb_side").is_not_null())
sample_b = r13b["match_key"].unique(maintain_order=True).head(4).to_list()
print("\n== (c) sample matches with round-13 bomb outcome, days 12-25")
for mk in sample_b:
    seq = (ALL.filter(pl.col("match_key") == mk).filter(pl.col("round").is_between(9, 16))
           .select(["round", "re_w", "reason", "sc_win", "bomb_side", "t_b", "c_b"]).sort("round"))
    print(mk)
    print(seq)

# (d) sample matches with round-13 score/round_end disagreement and no bomb outcome
dis_nb = (ALL.filter((pl.col("round") == 13) & (pl.col("day") >= 12) & pl.col("bomb_side").is_null()
                     & pl.col("sc_win").is_not_null() & (pl.col("sc_win") != pl.col("re_w"))))
sample_d = dis_nb["match_key"].unique(maintain_order=True).head(3).to_list()
print("\n== (d) sample matches with round-13 score/round_end disagreement, no bomb outcome, days 12-25")
for mk in sample_d:
    seq = (ALL.filter(pl.col("match_key") == mk).filter(pl.col("round").is_between(9, 16))
           .select(["round", "re_w", "reason", "sc_win", "bomb_side", "t_b", "c_b"]).sort("round"))
    print(mk)
    print(seq)

# (e) per-match round count and freeze count alignment, round 13 disagreement matches (days 12-25)
print("\n== (e) index alignment: rounds in round_end vs freeze_end per match (days 12-25)")
re_cnt = RE.filter(pl.col("player_count") == 20).group_by("match_key").agg(pl.col("round").n_unique().alias("re_rounds"),
                                                                       pl.col("round").max().alias("re_max"))
fe_cnt = fe.group_by("match_key").agg(pl.col("round").n_unique().alias("fe_rounds"), pl.col("round").max().alias("fe_max"))
al = re_cnt.join(fe_cnt, on="match_key", how="inner").with_columns(
    (pl.col("re_rounds") - pl.col("fe_rounds")).alias("diff"),
    (pl.col("re_max") - pl.col("fe_max")).alias("maxdiff"),
    pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).cast(pl.Int64).alias("day"))
print(al.filter(pl.col("day") >= 12).group_by(["diff", "maxdiff"]).len().sort("len", descending=True))

print("\n== DONE")
