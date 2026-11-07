"""verify D9-8 step 11: is the round_end timing stamp a match-level property that co-occurs with the flips?

Per match (2024-02, pc20):
  stamp_late(r)  : round_end tick is within 25 ticks of round_officially_ended (gap -19 or -20)
  stamp_early(r) : round_end tick is ~448 ticks before round_officially_ended (gap -447..-449)
  flip(r)        : score winner disagrees with round_end winner (round 27 excluded: oracle inverted there)
Q15 per match: how many late/early stamps; does the match carry a flip; split by day group
Q16 for round 13 only: flip rate by the stamp of that round, and by the match's dominant stamp
Q17 for bomb rounds: round_end minus bomb event tick, by match dominant stamp (sanity)
Output goes to stdout (tee'd by the caller). Scratch parquet only.
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(50)

SCRATCH = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
rst = pl.read_parquet(SCRATCH / "verify_D9_8_round_state.parquet")
re_all = pl.read_parquet(SCRATCH / "verify_D9_8_round_end.parquet").filter(pl.col("player_count") == 20)
sc = pl.read_parquet(SCRATCH / "verify_D9_8_score_winner.parquet")
rr = pl.read_parquet(SCRATCH / "verify_D9_8_rounds.parquet")
bs = pl.read_parquet(SCRATCH / "verify_D9_8_bomb_state.parquet")

re1 = re_all.group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("end_tick"),
                                                  pl.col("winner_team_code").first(),
                                                  pl.col("win_reason_message").first())
roe = (rst.filter(pl.col("event_type") == "round_officially_ended")
       .group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("roe_tick")))
M = (re1.join(roe, on=["match_key", "round"], how="inner")
     .join(sc.select(["match_key", "round", "score_winner"]), on=["match_key", "round"], how="left")
     .with_columns((pl.col("end_tick") - pl.col("roe_tick")).alias("gap"),
                   pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day")))
M = M.with_columns(pl.when(pl.col("day").cast(pl.Int32) <= 7).then(pl.lit("d01-07")).otherwise(pl.lit("d12-25")).alias("dg"),
                   pl.when(pl.col("gap").is_between(-25, -15)).then(pl.lit("late"))
                   .when(pl.col("gap").is_between(-460, -440)).then(pl.lit("early"))
                   .otherwise(pl.lit("other")).alias("stamp"),
                   ((pl.col("score_winner") != pl.col("winner_team_code")) & (pl.col("round") != 27)
                    & pl.col("score_winner").is_not_null()).alias("flip"))
print("stamp x day group (all pc20 rounds with a state gap):")
print(M.group_by(["dg", "stamp"]).agg(pl.len().alias("n"), pl.col("flip").sum().alias("flips")).sort(["dg", "stamp"]))

# Q15 per match
pm = (M.group_by(["match_key", "dg"]).agg(
    (pl.col("stamp") == "late").sum().alias("n_late"),
    (pl.col("stamp") == "early").sum().alias("n_early"),
    pl.len().alias("n_rounds"),
    pl.col("flip").sum().alias("n_flip")))
pm = pm.with_columns(pl.when(pl.col("n_late") > pl.col("n_early")).then(pl.lit("late_dominant"))
                     .when(pl.col("n_early") > 0).then(pl.lit("early_dominant")).otherwise(pl.lit("none")).alias("convention"))
print("\nQ15 matches by convention x day group, with flip counts:")
print(pm.group_by(["dg", "convention"]).agg(pl.len().alias("matches"), pl.col("n_flip").sum().alias("flips"),
                                             (pl.col("n_flip") > 0).sum().alias("matches_with_flip")).sort(["dg", "convention"]))

# Q16 round 13 flip rate by its own stamp and by match convention
r13 = M.filter(pl.col("round") == 13).join(pm.select(["match_key", "convention"]), on="match_key", how="left")
print("\nQ16 round 13 (scored): flip rate by own stamp, day group")
print(r13.filter(pl.col("score_winner").is_not_null()).group_by(["dg", "stamp"]).agg(
    pl.len().alias("n"), pl.col("flip").sum().alias("flips"), pl.col("flip").mean().round(3).alias("rate")).sort(["dg", "stamp"]))
print("\nQ16b round 13 (scored): flip rate by match convention, day group")
print(r13.filter(pl.col("score_winner").is_not_null()).group_by(["dg", "convention"]).agg(
    pl.len().alias("n"), pl.col("flip").sum().alias("flips"), pl.col("flip").mean().round(3).alias("rate")).sort(["dg", "convention"]))

# Q17 bomb rounds: round_end minus bomb tick by match convention
ev = (bs.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
      .group_by(["match_key", "round", "event_type"]).agg(pl.col("tick").min().alias("ev_tick")))
ev = ev.join(M.select(["match_key", "round", "end_tick", "dg", "flip", "stamp"]), on=["match_key", "round"], how="inner")
ev = ev.join(pm.select(["match_key", "convention"]), on="match_key", how="left")
ev = ev.with_columns((pl.col("end_tick") - pl.col("ev_tick")).alias("off"))
print("\nQ17 round_end minus bomb tick, by convention and flip (bomb rounds, pc20):")
print(ev.group_by(["dg", "convention", "flip"]).agg(pl.len().alias("n"), pl.col("off").median().alias("median_off")).sort(
    ["dg", "convention", "flip"]))
