"""verify D9-8 step 3: alternative explanations for the round-13 mismatch.

Hypotheses:
  H1 (claim): round_end record is bad (winner + message both wrong); bomb_state is right.
  H2 (alternative): bomb_state events are misattributed to round 13 (belong to another round);
      round_end is right.
  H3 (alternative): round_end and bomb_state are both right but the round label is shifted for
      the bomb events (off-by-one on bomb rows only).

Tests:
  T1 tick windows: is the defuse/explode tick inside the labelled round's [round_start tick,
     round_end tick] window? Also inside neighbour rounds 12 / 14.
  T2 round_state score deltas: does the team score move in the direction of the round_end winner
     or the bomb outcome across the labelled round? (independent of round_end and bomb_state)
  T3 plant tick inside the labelled window?

Reads the scratch parquet written by verify_D9_8_02_recompute.py (projected columns only), plus
round_state (projected) from the tome. Output goes to stdout (tee'd into logs\\swarm2\\verify_D9_8.log).
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(60)

ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
SCRATCH = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")


def files_of(ch):
    return sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))


rr = pl.read_parquet(SCRATCH / "verify_D9_8_rounds.parquet")
bs = pl.read_parquet(SCRATCH / "verify_D9_8_bomb_state.parquet")
re_all = pl.read_parquet(SCRATCH / "verify_D9_8_round_end.parquet")
rs_start = pl.read_parquet(SCRATCH / "verify_D9_8_round_start.parquet")

# round_end tick per (match, round) for player_count 20 AND all rows (use all for neighbours)
re_tick = re_all.select(["match_key", "round", "tick", "player_count"])
# the round window: start = round_start tick, end = round_end tick (player_count 20 rows)
win = (rs_start.select(["match_key", "round", pl.col("tick").alias("start_tick")])
       .join(re_all.filter(pl.col("player_count") == 20).select(["match_key", "round", pl.col("tick").alias("end_tick")]),
             on=["match_key", "round"], how="inner"))
print(f"round windows (round_start tick .. round_end tick, player_count 20): {win.height}")
print("windows with end_tick <= start_tick:", int((win["end_tick"] <= win["start_tick"]).sum()))

# ---- T0: monotonicity of round_end ticks inside each match (sanity of round numbering)
ordered = re_all.filter(pl.col("player_count") == 20).sort(["match_key", "round"])
ordered = ordered.with_columns(pl.col("tick").shift(1).over("match_key").alias("prev_tick"),
                               pl.col("round").shift(1).over("match_key").alias("prev_round"))
bad_order = ordered.filter(pl.col("tick") <= pl.col("prev_tick"))
print(f"\nT0 round_end ticks not increasing with round number (within match): {bad_order.height}")
if bad_order.height:
    print(bad_order.select(["match_key", "round", "prev_round", "tick", "prev_tick"]).head(10))

# ---- T1: bomb outcome tick vs labelled window
ev = (bs.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
      .group_by(["match_key", "round", "event_type"]).agg(pl.col("tick").min().alias("ev_tick")))
ev = ev.join(win, on=["match_key", "round"], how="left")
ev = ev.with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"))
ev = ev.with_columns(pl.when(pl.col("day").cast(pl.Int32) <= 7).then(pl.lit("d01-07")).otherwise(pl.lit("d12-25")).alias("dgrp"))

# neighbour windows
win_prev = win.with_columns((pl.col("round") + 1).alias("round")).select(
    ["match_key", "round", pl.col("start_tick").alias("prev_start"), pl.col("end_tick").alias("prev_end")])
win_next = win.with_columns((pl.col("round") - 1).alias("round")).select(
    ["match_key", "round", pl.col("start_tick").alias("next_start"), pl.col("end_tick").alias("next_end")])
# window of round r-1 (for an event labelled r, check whether it sits in r-1)
win_m1 = win.with_columns((pl.col("round") + 1).alias("round")).select(
    ["match_key", "round", pl.col("start_tick").alias("m1_start"), pl.col("end_tick").alias("m1_end")])
win_p1 = win.with_columns((pl.col("round") - 1).alias("round")).select(
    ["match_key", "round", pl.col("start_tick").alias("p1_start"), pl.col("end_tick").alias("p1_end")])
ev = ev.join(win_m1, on=["match_key", "round"], how="left").join(win_p1, on=["match_key", "round"], how="left")
ev = ev.with_columns(
    ((pl.col("ev_tick") >= pl.col("start_tick")) & (pl.col("ev_tick") <= pl.col("end_tick"))).alias("in_own"),
    ((pl.col("ev_tick") >= pl.col("m1_start")) & (pl.col("ev_tick") <= pl.col("m1_end"))).alias("in_r_minus_1"),
    ((pl.col("ev_tick") >= pl.col("p1_start")) & (pl.col("ev_tick") <= pl.col("p1_end"))).alias("in_r_plus_1"),
)
# join mismatch flag
ev = ev.join(rr.select(["match_key", "round", "mismatch", "generic"]), on=["match_key", "round"], how="left")
ev = ev.with_columns(
    pl.when(pl.col("in_own")).then(pl.lit("in_own_window"))
    .when(pl.col("in_r_minus_1")).then(pl.lit("in_round_minus1_window"))
    .when(pl.col("in_r_plus_1")).then(pl.lit("in_round_plus1_window"))
    .otherwise(pl.lit("none_of_three")).alias("where")
)
print("\nT1 outcome-event tick location, by mismatch flag and day group (round 13 only):")
t1 = ev.filter(pl.col("round") == 13)
print(t1.group_by(["dgrp", "mismatch", "event_type", "where"]).agg(pl.len().alias("n")).sort(
    ["dgrp", "mismatch", "event_type", "where"]))

print("\nT1 outcome-event tick location, all rounds, by mismatch flag and day group (control):")
t1o = ev.filter(pl.col("round") != 13)
print(t1o.group_by(["dgrp", "mismatch", "where"]).agg(pl.len().alias("n")).sort(["dgrp", "mismatch", "where"]))

# outcome event after round_end (tick > end_tick) or before start?
ev = ev.with_columns(
    pl.when(pl.col("ev_tick") > pl.col("end_tick")).then(pl.lit("after_end"))
    .when(pl.col("ev_tick") < pl.col("start_tick")).then(pl.lit("before_start"))
    .otherwise(pl.lit("inside")).alias("rel_own")
)
print("\nT1b relation of the outcome tick to its own round window (round 13, d12-25 only):")
print(ev.filter((pl.col("round") == 13) & (pl.col("dgrp") == "d12-25")).group_by(["mismatch", "event_type", "rel_own"])
      .agg(pl.len().alias("n")).sort(["mismatch", "event_type", "rel_own"]))

# ---- T3: planter tick location for round 13 mismatches
plant = (bs.filter(pl.col("event_type") == "bomb_planted")
         .group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("plant_tick")))
pt = rr.select(["match_key", "round", "mismatch", "dgrp" if "dgrp" in rr.columns else "day"]).join(
    plant, on=["match_key", "round"], how="left").join(win, on=["match_key", "round"], how="left")
pt = pt.filter(pl.col("round") == 13).with_columns(
    pl.when(pl.col("plant_tick").is_null()).then(pl.lit("no_plant"))
    .when((pl.col("plant_tick") >= pl.col("start_tick")) & (pl.col("plant_tick") <= pl.col("end_tick"))).then(pl.lit("inside"))
    .otherwise(pl.lit("outside")).alias("plant_loc"))
print("\nT3 plant tick vs own window, round 13 (rr rows):")
print(pt.group_by(["mismatch", "plant_loc"]).agg(pl.len().alias("n")).sort(["mismatch", "plant_loc"]))

# ---- T2: round_state score deltas across the labelled round
rst = (pl.scan_parquet(files_of("round_state")).select(
    ["match_key", "round", "tick", "event_type", "t_score", "ct_score", "t_score_raw", "ct_score_raw", "phase", "is_warmup"])
    .collect())
print(f"\nround_state rows: {rst.height}")
print("round_state event_type counts:")
print(rst.group_by("event_type").agg(pl.len().alias("n")).sort("event_type"))
print("round_state phase counts:")
print(rst.group_by("phase").agg(pl.len().alias("n")).sort("phase").head(20))
rst.write_parquet(SCRATCH / "verify_D9_8_round_state.parquet")
