"""D1-10 verifier: are the bomb events behind the 2024 round-13 mismatches inside their own round?

For each bomb_exploded / bomb_defused row in the tome (2026 and 2024), classify its tick relative to
the round's own window [round_start tick, round_end tick] and to the previous round's end tick.
If the 2024 round-13 'bomb exploded but CT won' rows fall before round 13 starts (i.e. they belong to
round 12's time window), the bomb evidence for round 13 is mis-attributed and is not a winner defect.

ASSUMPTIONS:
  A1 round window = [round_freeze_end tick from round_state, basic_round_end tick] for the same match and round.
  A2 a bomb event is 'in own window' when start_tick <= tick <= end_tick (boundaries inclusive).
  A3 'before own window' means tick < start_tick, and 'in previous round' means tick <= previous round end.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(12)

T26 = Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds")
T24 = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")


def fl(root, ch):
    return sorted(glob.glob(str(root / f"basic_{ch}*" / "dataframe_*")))


def run(root, label):
    # Anchor = round_freeze_end from round_state (live window start, as used by the bake-off).
    # The basic_round_start channel is NOT used: in the 2024 tome its tick for round r equals the
    # round_officially_ended tick of round r, i.e. the end of the round, so it is not a start anchor there.
    st = pl.concat([pl.scan_parquet(f).select(["match_key", "round", "tick", "event_type"]).collect() for f in fl(root, "round_state")])
    st = st.filter(pl.col("event_type") == "round_freeze_end").group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("start_tick"))
    en = pl.concat([pl.scan_parquet(f).select(["match_key", "round", "tick", "winner_team_code"]).collect() for f in fl(root, "round_end")])
    en = en.unique(["match_key", "round"]).rename({"tick": "end_tick"})
    prev_end = en.select(["match_key", (pl.col("round") + 1).alias("round"), pl.col("end_tick").alias("prev_end_tick")])
    bs = pl.concat([pl.scan_parquet(f).select(["match_key", "round", "tick", "event_type"]).collect() for f in fl(root, "bomb_state")])
    bs = bs.filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"]))
    j = (bs.join(st, on=["match_key", "round"], how="left")
           .join(en, on=["match_key", "round"], how="left")
           .join(prev_end, on=["match_key", "round"], how="left"))
    j = j.with_columns(
        pl.when(pl.col("start_tick").is_null() | pl.col("end_tick").is_null()).then(pl.lit("no_window"))
          .when((pl.col("tick") >= pl.col("start_tick")) & (pl.col("tick") <= pl.col("end_tick"))).then(pl.lit("in_own_round"))
          .when(pl.col("tick") < pl.col("start_tick")).then(pl.lit("before_own_round"))
          .otherwise(pl.lit("after_own_round")).alias("timing"),
        pl.when(pl.col("event_type") == "bomb_exploded").then(pl.lit(2)).otherwise(pl.lit(3)).alias("bomb_side"),
    )
    j = j.with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("bomb_dis"))
    print(f"\n== {label}: bomb outcome rows (exploded/defused) {j.height}; disagreeing with winner {int(j['bomb_dis'].sum())}")
    print("   timing counts (all rows):")
    for r in j.group_by("timing").len().sort("timing").iter_rows(named=True):
        print(f"     {r['timing']:<18} {r['len']}")
    print("   timing x disagreement:")
    for r in j.group_by(["timing", "bomb_dis"]).len().sort(["timing", "bomb_dis"]).iter_rows(named=True):
        print(f"     {r['timing']:<18} bomb_dis={r['bomb_dis']!s:<5} n={r['len']}")
    r13 = j.filter(pl.col("round") == 13)
    print(f"   round 13 only: rows {r13.height}, disagreeing {int(r13['bomb_dis'].sum())}")
    for r in r13.group_by(["timing", "bomb_dis"]).len().sort(["timing", "bomb_dis"]).iter_rows(named=True):
        print(f"     {r['timing']:<18} bomb_dis={r['bomb_dis']!s:<5} n={r['len']}")
    # rows where the bomb tick is inside the previous round's window (tick <= prev end and tick >= prev start)
    print("   round 13 disagreeing rows: tick minus own start (ticks), minus previous end (ticks), quantiles:")
    dis13 = r13.filter(pl.col("bomb_dis"))
    if dis13.height:
        d = dis13.with_columns((pl.col("tick") - pl.col("start_tick")).alias("dt_start"),
                               (pl.col("tick") - pl.col("prev_end_tick")).alias("dt_prev_end"))
        print("     dt_start:", d.select(pl.col("dt_start").quantile(0.05).alias("p05"), pl.col("dt_start").median().alias("p50"),
                                          pl.col("dt_start").quantile(0.95).alias("p95")).row(0))
        print("     dt_prev_end:", d.select(pl.col("dt_prev_end").quantile(0.05).alias("p05"), pl.col("dt_prev_end").median().alias("p50"),
                                             pl.col("dt_prev_end").quantile(0.95).alias("p95")).row(0))
    return j


j26 = run(T26, "2026")
j24 = run(T24, "2024")
print("\nDONE")
