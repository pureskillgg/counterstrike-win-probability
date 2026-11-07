"""verify D7-3: score-based round winner from round_state (t_score / ct_score), independent of round_end and bomb events.

For round r of a match:
  pre(r)  = (t_score, ct_score) on the round_state row for round r with event round_freeze_end (fallback: earliest row of round r)
  post(r) = (t_score, ct_score) on the first round_state row with tick >= round_end tick of round r (old labeling, before halftime swap)
  winner_score = T (2) if post.t == pre.t + 1 and post.ct == pre.ct; CT (3) if post.ct == pre.ct + 1 and post.t == pre.t; else none.
Also prints coverage of round_state for round_end rows, and compares winner_score to round_end winner_team_code and to bomb truth.
Read-only. Writes its per-round table to the session scratchpad only.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(240)
ROOT = Path(r"G:\pureskill\csds-cs2")
T24 = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")


def chan_dirs(base, ch):
    return sorted(p for p in base.glob(f"basic_{ch}.*") if p.is_dir())


def read_chan(base, ch, cols, where=None):
    parts = []
    for d in chan_dirs(base, ch):
        for f in sorted(glob.glob(str(d / "dataframe_*"))):
            lf = pl.scan_parquet(f).select(cols)
            if where is not None:
                lf = lf.filter(where)
            parts.append(lf.collect())
    return pl.concat(parts, how="vertical_relaxed")


R = pl.read_parquet(str(SCR / "verify_D7_3_rounds.parquet"))
R = R.with_columns(
    pl.when(pl.col("round") == 13).then(pl.lit("r13"))
    .when(pl.col("round") == 9).then(pl.lit("r9"))
    .when(pl.col("round") == 28).then(pl.lit("r28"))
    .when(pl.col("round") == 25).then(pl.lit("r25"))
    .otherwise(pl.lit("other")).alias("rbin"))
RS = read_chan(T24, "round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score", "phase"])
print("round_state rows:", RS.height, "| matches:", RS["match_key"].n_unique())

# pre-round score
pre = (RS.filter(pl.col("event_type") == "round_freeze_end")
       .group_by(["match_key", "round"]).agg(pl.col("t_score").first().alias("pre_t"), pl.col("ct_score").first().alias("pre_ct"),
                                              pl.col("tick").min().alias("freeze_tick")))
fallback = (RS.sort("tick").group_by(["match_key", "round"]).agg(pl.col("t_score").first().alias("fb_t"),
                                                                  pl.col("ct_score").first().alias("fb_ct")))
print("rounds with round_freeze_end row:", pre.height, "| rounds in round_state at all:", fallback.height)

# post-round score: first round_state row with tick >= round_end tick, same round label
RE_tick = R.select(["match_key", "round", "tick"]).rename({"tick": "end_tick"})
RS_sorted = RS.sort("tick")
post_rows = (RE_tick.sort("end_tick")
             .join_asof(RS_sorted.select(["match_key", "round", "tick", "t_score", "ct_score"]).rename({"round": "rs_round"}),
                        left_on="end_tick", right_on="tick", by="match_key", strategy="forward", tolerance=2000))
post_rows = post_rows.rename({"t_score": "post_t", "ct_score": "post_ct"})
print("round_end rows with a round_state row at/after end tick (within 2000 ticks):",
      post_rows.filter(pl.col("post_t").is_not_null()).height, "of", post_rows.height)
print("  of those, post row label == same round:", post_rows.filter(pl.col("rs_round") == pl.col("round")).height,
      "| label == round+1:", post_rows.filter(pl.col("rs_round") == pl.col("round") + 1).height)
post_rows = post_rows.unique(subset=["match_key", "round"], keep="first")

S = (R.join(pre.select(["match_key", "round", "pre_t", "pre_ct"]), on=["match_key", "round"], how="left")
     .join(fallback.select(["match_key", "round", "fb_t", "fb_ct"]), on=["match_key", "round"], how="left")
     .join(post_rows.select(["match_key", "round", "post_t", "post_ct"]), on=["match_key", "round"], how="left")
     .with_columns(pl.coalesce(["pre_t", "fb_t"]).alias("pre_t"), pl.coalesce(["pre_ct", "fb_ct"]).alias("pre_ct")))

S = S.with_columns(
    pl.when((pl.col("post_t") == pl.col("pre_t") + 1) & (pl.col("post_ct") == pl.col("pre_ct"))).then(pl.lit(2))
    .when((pl.col("post_ct") == pl.col("pre_ct") + 1) & (pl.col("post_t") == pl.col("pre_t"))).then(pl.lit(3))
    .otherwise(pl.lit(None)).cast(pl.Int64).alias("winner_score"),
)
S = S.with_columns(
    (pl.col("winner_score").is_not_null()).alias("has_score"),
    (pl.col("winner_score") != pl.col("winner_team_code")).alias("mm_score_vs_roundend"),
)
print("\nscore-based winner available for", S.filter(pl.col("has_score")).height, "of", S.height, "round_end rows")
print("score winner == round_end winner_team_code where both available:",
      S.filter(pl.col("has_score")).filter(pl.col("winner_score") == pl.col("winner_team_code")).height, "agree /",
      S.filter(pl.col("has_score")).height)

print("\n=== score-based vs round_end mismatch by build x round bin (ALL rounds with score) ===")
ss = S.filter(pl.col("has_score"))
print(ss.group_by(["build_num", "rbin"]).agg(pl.len().alias("n"), pl.col("mm_score_vs_roundend").sum().alias("mm"),
                                             pl.col("mm_score_vs_roundend").mean().alias("rate")).sort(["build_num", "rbin"]))

print("\n=== round 13 by day x build: score vs round_end ===")
r13s = ss.filter(pl.col("round") == 13)
print(r13s.group_by(["day", "build_num"]).agg(pl.len().alias("n"), pl.col("mm_score_vs_roundend").sum().alias("mm"),
                                              pl.col("mm_score_vs_roundend").mean().alias("rate")).sort(["day", "build_num"]))

print("\n=== score winner vs bomb truth (rounds with bomb outcome and score) by build x round bin ===")
bb = ss.filter(pl.col("has_bomb"))
bb = bb.with_columns((pl.col("winner_score") != pl.col("bomb_side")).alias("mm_score_vs_bomb"))
print(bb.group_by(["build_num", "rbin"]).agg(pl.len().alias("n"), pl.col("mm_score_vs_bomb").sum().alias("mm"),
                                             pl.col("mm_score_vs_bomb").mean().alias("rate")).sort(["build_num", "rbin"]))

print("\n=== score winner vs reason truth (rounds with score) by build x round bin ===")
print(ss.group_by(["build_num", "rbin"]).agg(
    pl.len().alias("n"), (pl.col("winner_score") != pl.col("reason_side")).sum().alias("mm_score_vs_reason")).sort(["build_num", "rbin"]))

print("\n=== round-13 rows with score winner, build 9961: score winner vs round_end winner, by has_bomb ===")
print(r13s.filter(pl.col("build_num") == 9961).group_by(["has_bomb"]).agg(
    pl.len().alias("n"), pl.col("mm_score_vs_roundend").sum().alias("mm")))

S.write_parquet(str(SCR / "verify_D7_3_score_rounds.parquet"))
print("\nwrote per-round score table to scratchpad")
