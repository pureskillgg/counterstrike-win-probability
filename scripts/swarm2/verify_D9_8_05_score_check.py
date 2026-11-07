"""verify D9-8 step 5: independent round winner from the score, and corrected tick windows.

Label convention (established in step 4): round_end(r) and round_state 'round_officially_ended'(r)
fire within ~19 ticks of each other, and the score in that event already includes round r.
So the labelled round r lies in the interval (round_end(r-1), round_end(r)].

ASSUMPTIONS:
  A1 score winner: a round is won by the side whose score (t_score for T, ct_score for CT) rises by 1
     between round_officially_ended(r-1) and round_officially_ended(r). t_score/ct_score are side-based
     and the tome's round_state carries them per tick (checked in step 5 by control agreement).
  A2 the first round (r=1) uses the score 0-0 as its previous value.
  A3 we use the first round_officially_ended row per (match, round).

Tests:
  T1 corrected tick windows for bomb outcome events and bomb_planted.
  T2 score winner vs round_end winner vs bomb-outcome side, split by mismatch flag and round bin.
  T3 control: score winner vs round_end winner over ALL rounds (agreement should be ~100% if A1 holds).

Reads scratch parquet written by steps 2 and 3. Output goes to stdout (tee'd by the caller).
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(50)

SCRATCH = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
rr = pl.read_parquet(SCRATCH / "verify_D9_8_rounds.parquet")          # round_end (pc20) x bomb outcome, with mismatch flag
re_all = pl.read_parquet(SCRATCH / "verify_D9_8_round_end.parquet")
bs = pl.read_parquet(SCRATCH / "verify_D9_8_bomb_state.parquet")
rst = pl.read_parquet(SCRATCH / "verify_D9_8_round_state.parquet")

# ---- corrected windows from round_end channel (player_count 20)
ree = re_all.filter(pl.col("player_count") == 20).select(["match_key", "round", pl.col("tick").alias("end_tick")])
ree = ree.group_by(["match_key", "round"]).agg(pl.col("end_tick").min())
prev = ree.with_columns((pl.col("round") + 1).alias("round")).select(
    ["match_key", "round", pl.col("end_tick").alias("prev_end_tick")])
win = ree.join(prev, on=["match_key", "round"], how="left")   # round r: (prev_end, end]
win = win.with_columns(pl.col("prev_end_tick").fill_null(0))

# ---- T1: outcome event tick and plant tick vs corrected window
ev = (bs.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded", "bomb_planted"]))
      .group_by(["match_key", "round", "event_type"]).agg(pl.col("tick").min().alias("ev_tick")))
ev = ev.join(win, on=["match_key", "round"], how="inner")
ev = ev.with_columns(
    ((pl.col("ev_tick") > pl.col("prev_end_tick")) & (pl.col("ev_tick") <= pl.col("end_tick"))).alias("inside"))
ev = ev.join(rr.select(["match_key", "round", "mismatch", "dgrp", "rbin"]), on=["match_key", "round"], how="left")
print("T1 outcome/plant events inside own round window (prev_end, end], round 13 only:")
print(ev.filter(pl.col("round") == 13).group_by(["dgrp", "mismatch", "event_type", "inside"]).agg(pl.len().alias("n"))
      .sort(["dgrp", "mismatch", "event_type", "inside"]))
print("\nT1 control, rounds != 13:")
print(ev.filter(pl.col("round") != 13).group_by(["dgrp", "event_type", "inside"]).agg(pl.len().alias("n"))
      .sort(["dgrp", "event_type", "inside"]))

# ---- T2/T3: score-based winner
so = (rst.filter(pl.col("event_type") == "round_officially_ended")
      .group_by(["match_key", "round"]).agg(pl.col("t_score").first(), pl.col("ct_score").first(),
                                            pl.col("t_score_raw").first(), pl.col("ct_score_raw").first()))
print(f"\nround_officially_ended (match, round) rows: {so.height}")
prev_so = so.with_columns((pl.col("round") + 1).alias("round")).select(
    ["match_key", "round", pl.col("t_score").alias("t_prev"), pl.col("ct_score").alias("ct_prev"),
     pl.col("t_score_raw").alias("tr_prev"), pl.col("ct_score_raw").alias("cr_prev")])
sc = so.join(prev_so, on=["match_key", "round"], how="left")
sc = sc.with_columns(
    pl.when(pl.col("t_prev").is_null() & (pl.col("round") == 1)).then(0).otherwise(pl.col("t_prev")).alias("t_prev"),
    pl.when(pl.col("ct_prev").is_null() & (pl.col("round") == 1)).then(0).otherwise(pl.col("ct_prev")).alias("ct_prev"),
)
sc = sc.with_columns(
    (pl.col("t_score") - pl.col("t_prev")).alias("dT"),
    (pl.col("ct_score") - pl.col("ct_prev")).alias("dCT"),
).with_columns(
    pl.when((pl.col("dT") == 1) & (pl.col("dCT") == 0)).then(2)
    .when((pl.col("dCT") == 1) & (pl.col("dT") == 0)).then(3)
    .otherwise(None).alias("score_winner"))
print("score_winner availability and delta patterns (all rounds with round_officially_ended):")
print(sc.group_by(["dT", "dCT"]).agg(pl.len().alias("n")).sort(["dT", "dCT"]))

# join to the bomb-outcome round table
j = rr.join(sc.select(["match_key", "round", "score_winner", "dT", "dCT"]), on=["match_key", "round"], how="left")
print(f"\nbomb-outcome rounds (pc20) joined: {j.height}; with score_winner: {int(j['score_winner'].is_not_null().sum())}")
print("T2 score_winner vs round_end winner vs bomb side, by day group x round bin x mismatch flag:")
t2 = (j.filter(pl.col("score_winner").is_not_null())
      .with_columns(
          (pl.col("score_winner") == pl.col("winner_team_code")).alias("score_eq_roundend"),
          (pl.col("score_winner") == pl.col("bomb_side")).alias("score_eq_bomb"))
      .group_by(["dgrp", "rbin", "mismatch"])
      .agg(pl.len().alias("n"),
           pl.col("score_eq_roundend").sum().alias("score_eq_roundend"),
           pl.col("score_eq_bomb").sum().alias("score_eq_bomb"))
      .sort(["dgrp", "rbin", "mismatch"]))
print(t2)

print("\nT2b round 13 mismatch rows (d12-25): score_winner distribution and agreement")
m13 = j.filter((pl.col("round") == 13) & pl.col("mismatch"))
print(f"  mismatch rows: {m13.height}, with score_winner: {int(m13['score_winner'].is_not_null().sum())}")
print("  score_winner x round_end winner x bomb side:")
print(m13.group_by(["score_winner", "winner_team_code", "bomb_side"]).agg(pl.len().alias("n")).sort(
    ["score_winner", "winner_team_code", "bomb_side"]))
print("  message x score_winner:")
print(m13.group_by(["win_reason_message", "score_winner"]).agg(pl.len().alias("n")).sort(
    ["win_reason_message", "score_winner"]))

# T3 control: over ALL rounds (not only bomb rounds) agreement score vs round_end
allr = sc.join(re_all.filter(pl.col("player_count") == 20).group_by(["match_key", "round"]).agg(
    pl.col("winner_team_code").first()), on=["match_key", "round"], how="inner")
allr = allr.filter(pl.col("score_winner").is_not_null())
print(f"\nT3 control: all pc20 rounds with a clean score delta: {allr.height}")
print("  score_winner == round_end winner:", int((allr["score_winner"] == allr["winner_team_code"]).sum()),
      "  disagree:", int((allr["score_winner"] != allr["winner_team_code"]).sum()))
allr = allr.with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"))
allr = allr.with_columns(pl.when(pl.col("day").cast(pl.Int32) <= 7).then(pl.lit("d01-07")).otherwise(pl.lit("d12-25")).alias("dgrp"))
print(allr.group_by(["dgrp", "round"]).agg(
    pl.len().alias("n"),
    (pl.col("score_winner") != pl.col("winner_team_code")).sum().alias("disagree")).filter(pl.col("round").is_in([11, 12, 13, 14, 15])).sort(["dgrp", "round"]))

sc.write_parquet(SCRATCH / "verify_D9_8_score_winner.parquet")
