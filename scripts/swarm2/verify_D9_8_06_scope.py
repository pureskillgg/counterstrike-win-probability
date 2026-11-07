"""verify D9-8 step 6: scope of the score-vs-round_end disagreement, and explosion timing.

Uses scratch parquet from steps 2, 3 and 5. Questions:
  Q1 Across ALL rounds (bomb or not), where does score_winner disagree with round_end winner?
     Split by round number and day group (is it round-13-only and d12-25-only, or wider?).
  Q2 Among round-13 d12-25 disagreements that have NO bomb outcome, what are the messages and
     the direction of the flip (score says T vs CT)?
  Q3 Timing of bomb_exploded relative to round_end (ev_tick - end_tick) in control rounds, to
     calibrate the "outside window" cases in step 5.
  Q4 Round 13 flip rate by day (d12-25), all rounds, to see whether the defect is day-dependent.
Output goes to stdout (tee'd by the caller).
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(120)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(50)

SCRATCH = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
re_all = pl.read_parquet(SCRATCH / "verify_D9_8_round_end.parquet").filter(pl.col("player_count") == 20)
sc = pl.read_parquet(SCRATCH / "verify_D9_8_score_winner.parquet")
bs = pl.read_parquet(SCRATCH / "verify_D9_8_bomb_state.parquet")
rr = pl.read_parquet(SCRATCH / "verify_D9_8_rounds.parquet")

re1 = (re_all.group_by(["match_key", "round"])
       .agg(pl.col("winner_team_code").first(), pl.col("win_reason_message").first(), pl.col("tick").min().alias("end_tick")))
re1 = re1.with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"))
re1 = re1.with_columns(pl.when(pl.col("day").cast(pl.Int32) <= 7).then(pl.lit("d01-07")).otherwise(pl.lit("d12-25")).alias("dgrp"))

A = re1.join(sc.select(["match_key", "round", "score_winner", "dT", "dCT"]), on=["match_key", "round"], how="left")
A = A.with_columns(pl.col("score_winner").is_not_null().alias("has_score"),
                   (pl.col("score_winner") != pl.col("winner_team_code")).alias("disagree"))
scored = A.filter(pl.col("has_score"))
print(f"round_end (pc20) rounds: {A.height}; with clean score winner: {scored.height}; disagree: {int(scored['disagree'].sum())}")

print("\nQ1 disagreements by day group x round number (all rounds with clean score winner), rounds with any disagreement:")
q1 = (scored.group_by(["dgrp", "round"]).agg(pl.len().alias("n"), pl.col("disagree").sum().alias("disagree"))
      .with_columns((pl.col("disagree") / pl.col("n")).round(3).alias("rate")))
print(q1.filter(pl.col("disagree") > 0).sort(["dgrp", "round"]))
print("\nQ1b total by day group:")
print(scored.group_by("dgrp").agg(pl.len().alias("n"), pl.col("disagree").sum().alias("disagree")).sort("dgrp"))
print("\nQ1c disagreements outside round 13, by day group:")
print(scored.filter(pl.col("round") != 13).group_by("dgrp").agg(pl.len().alias("n"), pl.col("disagree").sum().alias("disagree")).sort("dgrp"))
print("\nQ1d disagreements outside round 13, by round number (top rounds):")
print(scored.filter((pl.col("round") != 13) & pl.col("disagree")).group_by(["dgrp", "round"]).agg(pl.len().alias("n")).sort(["dgrp", "round"]).head(60))

# Q2: round 13, d12-25, disagreement with and without bomb outcome
bombkey = rr.select(["match_key", "round"]).with_columns(pl.lit(True).alias("has_bomb"))
r13 = scored.filter((pl.col("round") == 13) & (pl.col("dgrp") == "d12-25")).join(bombkey, on=["match_key", "round"], how="left")
r13 = r13.with_columns(pl.col("has_bomb").fill_null(False))
print("\nQ2 round 13, d12-25, scored rounds: disagreement by bomb-outcome presence")
print(r13.group_by(["has_bomb", "disagree"]).agg(pl.len().alias("n")).sort(["has_bomb", "disagree"]))
print("\nQ2b messages on round-13 d12-25 disagreements, by bomb presence")
print(r13.filter(pl.col("disagree")).group_by(["has_bomb", "win_reason_message", "score_winner", "winner_team_code"])
      .agg(pl.len().alias("n")).sort(["has_bomb", "win_reason_message"]))
print("\nQ2c round-13 d12-25 messages on AGREEING rounds (control), by bomb presence")
print(r13.filter(~pl.col("disagree")).group_by(["has_bomb", "win_reason_message"]).agg(pl.len().alias("n")).sort(["has_bomb", "win_reason_message"]))

# Q4 day profile of round-13 disagreement rate (all round-13 rounds with clean score)
print("\nQ4 round 13 disagreement rate by day (score vs round_end):")
r13all = scored.filter(pl.col("round") == 13)
print(r13all.group_by("day").agg(pl.len().alias("n"), pl.col("disagree").sum().alias("disagree"),
                                  pl.col("disagree").mean().round(3).alias("rate")).sort("day"))

# Q3 explosion timing relative to round_end (control rounds and round 13)
ex = (bs.filter(pl.col("event_type") == "bomb_exploded").group_by(["match_key", "round"])
      .agg(pl.col("tick").min().alias("ex_tick")))
ex = ex.join(re1.select(["match_key", "round", "end_tick", "dgrp", "winner_team_code"]), on=["match_key", "round"], how="inner")
ex = ex.with_columns((pl.col("ex_tick") - pl.col("end_tick")).alias("ex_minus_end"))
print("\nQ3 bomb_exploded tick minus round_end tick (control: round != 13), quantiles by day group:")
ctrl = ex.filter(pl.col("round") != 13)
for dg in ["d01-07", "d12-25"]:
    s = ctrl.filter(pl.col("dgrp") == dg)["ex_minus_end"]
    print(dg, "n=", s.len(), "min=", s.min(), "p05=", s.quantile(0.05), "median=", s.median(),
          "p95=", s.quantile(0.95), "max=", s.max(), "share>0=", round(float((s > 0).mean()), 3))
print("Q3b round 13 exploded, ex_minus_end distribution by winner code:")
r13x = ex.filter(pl.col("round") == 13)
print(r13x.group_by(["dgrp", "winner_team_code"]).agg(pl.len().alias("n"),
                                                       pl.col("ex_minus_end").median().alias("median_ex_minus_end")).sort(["dgrp", "winner_team_code"]))
