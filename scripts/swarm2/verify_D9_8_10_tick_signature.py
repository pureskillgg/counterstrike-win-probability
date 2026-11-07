"""verify D9-8 step 10: tick-offset signature of the defective round_end records, and 2026 OT swap rounds.

Q11 tick offset: round_end tick minus bomb-outcome tick (defused or exploded), split by day group,
    round bin, mismatch flag and winner code. Tests whether the defective records carry a distinct
    timing signature (independent of the win message).
Q12 non-bomb round 13 (d12-25): round_end tick minus round_officially_ended tick (state event), split by
    agreement with the score winner. Tests whether non-bomb flips have a timing signature too.
Q13 messages on 2024 rounds 25 and 28 (d12-25) score disagreements (all generic?).
Q14 2026 score disagreements (rounds 27/33/39/45): bomb-outcome side vs round_end vs score, where a bomb exists.
Reads scratch parquet from steps 2, 3, 5, 9. Output goes to stdout (tee'd by the caller).
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(50)

SCRATCH = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
rr = pl.read_parquet(SCRATCH / "verify_D9_8_rounds.parquet")
bs = pl.read_parquet(SCRATCH / "verify_D9_8_bomb_state.parquet")
re_all = pl.read_parquet(SCRATCH / "verify_D9_8_round_end.parquet").filter(pl.col("player_count") == 20)
rst = pl.read_parquet(SCRATCH / "verify_D9_8_round_state.parquet")
sc = pl.read_parquet(SCRATCH / "verify_D9_8_score_winner.parquet")

re1 = re_all.group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("end_tick"),
                                                  pl.col("winner_team_code").first(),
                                                  pl.col("win_reason_message").first())

# Q11: outcome tick per (match, round, event)
ev = (bs.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
      .group_by(["match_key", "round", "event_type"]).agg(pl.col("tick").min().alias("ev_tick")))
ev = ev.join(re1, on=["match_key", "round"], how="inner").join(
    rr.select(["match_key", "round", "mismatch", "dgrp"]), on=["match_key", "round"], how="left")
ev = ev.with_columns((pl.col("end_tick") - pl.col("ev_tick")).alias("off"),
                     pl.when(pl.col("round") == 13).then(pl.lit("r13")).otherwise(pl.lit("other")).alias("rbin"))
print("Q11 round_end tick minus outcome tick, by event, day group, round bin, winner, mismatch:")
q11 = (ev.group_by(["event_type", "dgrp", "rbin", "winner_team_code", "mismatch"])
       .agg(pl.len().alias("n"), pl.col("off").median().alias("median_off"),
            pl.col("off").min().alias("min_off"), pl.col("off").max().alias("max_off"))
       .sort(["event_type", "dgrp", "rbin", "winner_team_code", "mismatch"]))
print(q11)

# Q12: non-bomb round 13 d12-25: round_end tick vs round_officially_ended tick
roe = (rst.filter(pl.col("event_type") == "round_officially_ended")
       .group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("roe_tick")))
bomb_keys = rr.select(["match_key", "round"]).unique()
nb = (re1.join(roe, on=["match_key", "round"], how="inner")
      .join(sc.select(["match_key", "round", "score_winner"]), on=["match_key", "round"], how="inner")
      .join(bomb_keys.with_columns(pl.lit(True).alias("has_bomb")), on=["match_key", "round"], how="left")
      .with_columns(pl.col("has_bomb").fill_null(False),
                    (pl.col("score_winner") != pl.col("winner_team_code")).alias("disagree"),
                    pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day")))
nb = nb.filter((pl.col("round") == 13) & (pl.col("day").cast(pl.Int32) >= 12) & (~pl.col("has_bomb")))
nb = nb.with_columns((pl.col("end_tick") - pl.col("roe_tick")).alias("end_minus_roe"))
print(f"\nQ12 non-bomb round 13 d12-25 scored rounds: {nb.height}")
print(nb.group_by("disagree").agg(pl.len().alias("n"), pl.col("end_minus_roe").median().alias("median"),
                                  pl.col("end_minus_roe").min().alias("min"), pl.col("end_minus_roe").max().alias("max"),
                                  pl.col("end_minus_roe").quantile(0.05).alias("p05"),
                                  pl.col("end_minus_roe").quantile(0.95).alias("p95")).sort("disagree"))
print(nb.group_by(["disagree", "end_minus_roe"]).agg(pl.len().alias("n")).sort("n", descending=True).head(12))

# Q13: messages on round 25 / 28 disagreements (2024 d12-25)
print("\nQ13 messages on score disagreements, d12-25, rounds 25 and 28:")
x = (re1.join(sc.select(["match_key", "round", "score_winner"]), on=["match_key", "round"], how="inner")
     .with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day")))
x = x.filter((pl.col("day").cast(pl.Int32) >= 12) & pl.col("round").is_in([9, 13, 25, 28]) & pl.col("score_winner").is_not_null())
x = x.with_columns((pl.col("score_winner") != pl.col("winner_team_code")).alias("disagree"))
print(x.filter(pl.col("disagree")).group_by(["round", "win_reason_message"]).agg(pl.len().alias("n")).sort(["round", "win_reason_message"]))

# Q14: 2026 disagreements (rounds 27/33/39/45): compare with bomb side where present
J = pl.read_parquet(SCRATCH / "verify_D9_8_2026_rounds.parquet")
d26 = J.filter(pl.col("has_score") & pl.col("score_disagree"))
print(f"\nQ14 2026 score disagreements: {d26.height}")
print(d26.group_by(["round", "has_bomb"]).agg(pl.len().alias("n")).sort(["round", "has_bomb"]))
db = d26.filter(pl.col("has_bomb"))
print(f"2026 disagreements with a bomb outcome: {db.height}")
if db.height:
    print("  bomb==round_end:", int((db["bomb_side"] == db["winner_team_code"]).sum()),
          "  bomb==score:", int((db["bomb_side"] == db["score_winner"]).sum()))
print("2026 messages on disagreements:")
print(d26.group_by("win_reason_message").agg(pl.len().alias("n")).sort("n", descending=True))
