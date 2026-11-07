"""verify D9-8 step 12: freeze-to-freeze consistency test on the 2024 tome (method-independent of the ROE oracle).

Rule (copied from scripts\\model\\bakeoff.py swap_expected, re-implemented not imported):
  swap after completed round r  iff  r == max_rounds//2  or  (r > max_rounds and (r - max_rounds - 3) % 6 == 0)
max_rounds is inferred per match from round_state phase: halftime round H = (first round whose freeze_end
phase is 'second') - 1; max_rounds = 2*H  (ASSUMPTION A4; H is 12 for MR12).

For each consecutive pair (r, r+1) with round_end(r) and round_state freeze_end(r) and freeze_end(r+1):
  predicted next (t, ct) from round_end(r) winner:
      post = (t_r + [w==T], ct_r + [w==CT]);  pred = post if no swap, else (post_ct, post_t)
  consistent  iff pred == actual freeze_end(r+1)
  alt_consistent_with_flip: the same test with the winner inverted (what a flipped round_end would need)
Reports consistency by day group x round number, and whether the inconsistent set overlaps the defect rows.

Output goes to stdout (tee'd by the caller). Scratch parquet only.
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(50)

SCRATCH = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
rst = pl.read_parquet(SCRATCH / "verify_D9_8_round_state.parquet")
re_all = pl.read_parquet(SCRATCH / "verify_D9_8_round_end.parquet").filter(pl.col("player_count") == 20)
rr = pl.read_parquet(SCRATCH / "verify_D9_8_rounds.parquet")

re1 = re_all.group_by(["match_key", "round"]).agg(pl.col("winner_team_code").first(), pl.col("win_reason_message").first())
fz = (rst.filter(pl.col("event_type") == "round_freeze_end")
      .group_by(["match_key", "round"]).agg(pl.col("t_score").first(), pl.col("ct_score").first(),
                                            pl.col("phase").first()))
# halftime round per match: first freeze_end round whose phase is 'second', minus one
ht = (fz.filter(pl.col("phase") == "second").group_by("match_key").agg((pl.col("round").min() - 1).alias("H")))
print(f"matches with a 'second' half phase: {ht.height}")
print("H (halftime round) distribution:", ht.group_by("H").agg(pl.len().alias("matches")).sort("H").rows())
ht = ht.with_columns((2 * pl.col("H")).alias("max_rounds"))

fz = fz.join(ht, on="match_key", how="left")
# transitions
nxt = fz.with_columns((pl.col("round") - 1).alias("round")).select(
    ["match_key", "round", pl.col("t_score").alias("t_next"), pl.col("ct_score").alias("ct_next")])
T = fz.join(re1, on=["match_key", "round"], how="inner").join(nxt, on=["match_key", "round"], how="inner")
T = T.with_columns(
    pl.when(pl.col("max_rounds").is_null()).then(pl.lit(False))
    .when(pl.col("round") == pl.col("H")).then(pl.lit(True))
    .when((pl.col("round") > pl.col("max_rounds")) & (((pl.col("round") - pl.col("max_rounds") - 3) % 6) == 0)).then(pl.lit(True))
    .otherwise(pl.lit(False)).alias("swap"),
    pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"),
)
T = T.with_columns(pl.when(pl.col("day").cast(pl.Int32) <= 7).then(pl.lit("d01-07")).otherwise(pl.lit("d12-25")).alias("dg"))
T = T.with_columns(
    (pl.col("t_score") + (pl.col("winner_team_code") == 2).cast(pl.Int64)).alias("t_post"),
    (pl.col("ct_score") + (pl.col("winner_team_code") == 3).cast(pl.Int64)).alias("ct_post"),
)
T = T.with_columns(
    pl.when(pl.col("swap")).then(pl.col("ct_post")).otherwise(pl.col("t_post")).alias("pred_t"),
    pl.when(pl.col("swap")).then(pl.col("t_post")).otherwise(pl.col("ct_post")).alias("pred_ct"),
)
T = T.with_columns(((pl.col("pred_t") == pl.col("t_next")) & (pl.col("pred_ct") == pl.col("ct_next"))).alias("consistent"))
print(f"\ntransitions evaluated: {T.height}; consistent: {int(T['consistent'].sum())}; "
      f"inconsistent: {int((~T['consistent']).sum())}")

print("\nconsistency by day group (all rounds):")
print(T.group_by("dg").agg(pl.len().alias("n"), (~pl.col("consistent")).sum().alias("inconsistent")).sort("dg"))

print("\ninconsistent transitions by round number (round r is the completed round, transition r -> r+1):")
inc = T.filter(~pl.col("consistent"))
print(inc.group_by(["dg", "round"]).agg(pl.len().alias("n")).sort(["dg", "round"]).head(60))

# does the inconsistency mean the round_end winner is wrong? test: would inverting the winner restore consistency?
T = T.with_columns(
    (pl.col("t_score") + (pl.col("winner_team_code") == 3).cast(pl.Int64)).alias("t_post_inv"),
    (pl.col("ct_score") + (pl.col("winner_team_code") == 2).cast(pl.Int64)).alias("ct_post_inv"),
)
T = T.with_columns(
    pl.when(pl.col("swap")).then(pl.col("ct_post_inv")).otherwise(pl.col("t_post_inv")).alias("pred_t_inv"),
    pl.when(pl.col("swap")).then(pl.col("t_post_inv")).otherwise(pl.col("ct_post_inv")).alias("pred_ct_inv"),
)
T = T.with_columns(((pl.col("pred_t_inv") == pl.col("t_next")) & (pl.col("pred_ct_inv") == pl.col("ct_next"))).alias("consistent_if_inverted"))
print("\ninconsistent rounds: how many become consistent if the round_end winner is inverted (flip test):")
x = T.filter(~pl.col("consistent"))
print(x.group_by(["dg", "consistent_if_inverted"]).agg(pl.len().alias("n")).sort(["dg", "consistent_if_inverted"]))
print("\nconsistent_if_inverted among inconsistent rounds, by round:")
print(x.group_by(["dg", "round", "consistent_if_inverted"]).agg(pl.len().alias("n")).sort(["dg", "round"]).head(40))

# overlap with the bomb-based defect set (round 13, d12-25 mismatch) and with flip-test positives
mm = rr.filter(pl.col("mismatch")).select(["match_key", "round"]).with_columns(pl.lit(True).alias("bomb_mismatch"))
x2 = x.join(mm, on=["match_key", "round"], how="left").with_columns(pl.col("bomb_mismatch").fill_null(False))
print("\noverlap: inconsistent rounds that are bomb-based mismatches:")
print(x2.group_by(["round", "bomb_mismatch"]).agg(pl.len().alias("n")).sort(["round", "bomb_mismatch"]).head(20))

# control: consistent round-13 transitions in d12-25 by message
print("\nround 13, d12-25: consistency by round_end message")
print(T.filter((pl.col("round") == 13) & (pl.col("dg") == "d12-25")).group_by(["win_reason_message", "consistent"]).agg(
    pl.len().alias("n")).sort(["win_reason_message", "consistent"]))
T.write_parquet(SCRATCH / "verify_D9_8_freeze_transitions.parquet")
