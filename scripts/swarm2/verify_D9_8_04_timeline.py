"""verify D9-8 step 4: label-convention check on round_start / round_end / bomb events.

Question: in this tome, does round_start(r) mark the start of round r, or the start of round r+1
(i.e. is the round_start channel offset from round_end)? Establishes the correct round window.

Checks:
  C1 distribution of (round_start(r).tick - round_end(r).tick) and (round_start(r).tick - round_end(r-1).tick)
  C2 timeline of one defect match (round 11..15) with round_end, round_start, bomb events, round_state
     round events (cs_round_start_beep, round_freeze_end, round_end)
Reads the scratch parquet from step 2 and the round_state parquet from step 3.
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(50)

SCRATCH = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
re_all = pl.read_parquet(SCRATCH / "verify_D9_8_round_end.parquet").filter(pl.col("player_count") == 20)
rs = pl.read_parquet(SCRATCH / "verify_D9_8_round_start.parquet")
bs = pl.read_parquet(SCRATCH / "verify_D9_8_bomb_state.parquet")
rst = pl.read_parquet(SCRATCH / "verify_D9_8_round_state.parquet")

# C1: offsets between round_start(r) and round_end(r) / round_end(r-1)
ree = re_all.select(["match_key", "round", pl.col("tick").alias("re_tick")])
j0 = rs.select(["match_key", "round", pl.col("tick").alias("rs_tick")]).join(ree, on=["match_key", "round"], how="inner")
j0 = j0.with_columns((pl.col("rs_tick") - pl.col("re_tick")).alias("d_same"))
ree_prev = ree.with_columns((pl.col("round") + 1).alias("round")).select(
    ["match_key", "round", pl.col("re_tick").alias("re_prev_tick")])
j1 = rs.select(["match_key", "round", pl.col("tick").alias("rs_tick")]).join(ree_prev, on=["match_key", "round"], how="inner")
j1 = j1.with_columns((pl.col("rs_tick") - pl.col("re_prev_tick")).alias("d_prev"))
print("C1a rs(r) - re(r): quantiles")
print(j0["d_same"].describe())
print("\nC1b rs(r) - re(r-1): quantiles")
print(j1["d_prev"].describe())

# check round_state round_start / round_end events against round_end channel
rst_re = rst.filter(pl.col("event_type") == "round_end").select(["match_key", "round", pl.col("tick").alias("rse_tick")])
rst_rs = rst.filter(pl.col("event_type") == "round_start").select(["match_key", "round", pl.col("tick").alias("rss_tick")])
print(f"\nround_state round_end event rows: {rst_re.height}; round_start event rows: {rst_rs.height}")
print(f"round_state round_end matched to round_end channel (match, round, tick equal): "
      f"{rst_re.join(ree, on=['match_key','round'], how='inner').filter(pl.col('rse_tick') == pl.col('re_tick')).height}")

# pick a defect match: a match with round 13 mismatch
rr = pl.read_parquet(SCRATCH / "verify_D9_8_rounds.parquet")
mm = rr.filter((pl.col("round") == 13) & pl.col("mismatch")).select("match_key").unique().sort("match_key")
print(f"\ndefect matches (round 13 mismatch): {mm.height}")
sample_keys = mm["match_key"].head(3).to_list()
print("sample keys:", sample_keys)

for key in sample_keys[:1]:
    print(f"\n==== timeline for {key}")
    rounds = list(range(11, 16))
    t_re = re_all.filter((pl.col("match_key") == key) & pl.col("round").is_in(rounds)).select(
        ["round", "tick", "winner_team_code", "win_reason_message"]).sort("round")
    print("round_end channel:")
    print(t_re)
    t_rs = rs.filter((pl.col("match_key") == key) & pl.col("round").is_in(rounds)).select(
        ["round", "tick"]).sort("round")
    print("round_start channel:")
    print(t_rs)
    t_b = bs.filter((pl.col("match_key") == key) & pl.col("round").is_in(rounds)
                    & pl.col("event_type").is_in(["bomb_planted", "bomb_defused", "bomb_exploded"])).select(
        ["round", "tick", "event_type", "player_team_code"]).sort(["tick"]).unique(maintain_order=True)
    print("bomb events (first row per round/event/tick):")
    print(t_b)
    t_s = rst.filter((pl.col("match_key") == key) & pl.col("round").is_in(rounds)
                     & pl.col("event_type").is_in(["round_start", "round_end", "round_freeze_end", "round_officially_ended",
                                                  "cs_round_start_beep", "round_prestart"])).select(
        ["round", "tick", "event_type", "t_score", "ct_score"]).sort(["tick"])
    print("round_state round events:")
    print(t_s)
