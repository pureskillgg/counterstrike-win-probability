"""verify D9-9, step 3: scoreboard-based round winner vs round_end winner_team_code (2024-02 tome).

Independent source: round_state score fields. t_score and ct_score are side-based (they swap at
halftime, see the probe). For each (match, round) the winner by scoreboard is the side whose score
field increased between its round_freeze_end row (pre) and its round_end row (post). The round_end
table is compared against it, per build and per round index.

Reads round_state, round_end and header only (projection, lazy scan). Per-match rows are kept small.

ASSUMPTIONS (ASSUMPTION):
  - pre = round_state row with event_type round_freeze_end and the same round number; post = row with
    event_type round_end and the same round number (same match).
  - side codes: 2 = T, 3 = CT (same convention as round_end.winner_team_code).
  - the score fields are side-based: a T-side win increments t_score, a CT-side win increments ct_score.
    This is checked, not assumed, by the agreement rate outside the swap rounds.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(24)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"


def files_of(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


def scan(name, want):
    fd = BASE / "header" if name == "header" else BASE / f"basic_{name}{SUF}"
    lf = pl.scan_parquet(files_of(fd))
    have = set(lf.collect_schema().names())
    return lf.select([c for c in want if c in have]).collect()


H = scan("header", ["key", "build_num"]).rename({"key": "match_key"})
RS = scan("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
RE = scan("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_message",
                        "player_count"])

pre = RS.filter(pl.col("event_type") == "round_freeze_end").select(
    ["match_key", "round", pl.col("tick").alias("pre_tick"), pl.col("t_score").alias("t_pre"),
     pl.col("ct_score").alias("ct_pre")])
# post-round scores, pre-swap: 'cs_pre_restart' exists in both builds at the round_end tick
# (in build 9961 there is no 'round_end' state row); 'round_end' state rows are used as a check.
post_cpr = RS.filter(pl.col("event_type") == "cs_pre_restart").select(
    ["match_key", "round", pl.col("tick").alias("post_tick"), pl.col("t_score").alias("t_post"),
     pl.col("ct_score").alias("ct_post")])
post_re = RS.filter(pl.col("event_type") == "round_end").select(
    ["match_key", "round", pl.col("tick").alias("post_re_tick"), pl.col("t_score").alias("t_post_re"),
     pl.col("ct_score").alias("ct_post_re")])
print("pre rows", pre.height, "unique (match,round)", pre.select(["match_key", "round"]).unique().height)
print("cs_pre_restart rows", post_cpr.height, "unique (match,round)",
      post_cpr.select(["match_key", "round"]).unique().height)
print("round_state round_end rows", post_re.height, "| round_end table rows", RE.height,
      "unique (match,round)", RE.select(["match_key", "round"]).unique().height)

pre_d = pre.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1).height
cpr_d = post_cpr.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1).height
print("duplicate pre groups:", pre_d, "| duplicate cs_pre_restart groups:", cpr_d)
pre = pre.unique(subset=["match_key", "round"], keep="first")
post = post_cpr.unique(subset=["match_key", "round"], keep="last")
# check: cs_pre_restart tick equals round_end table tick for the same round (same round end)
tick_chk = (post.join(RE.select(["match_key", "round", pl.col("tick").alias("re_tick")]).unique(
    subset=["match_key", "round"], keep="first"), on=["match_key", "round"], how="inner")
    .with_columns((pl.col("post_tick") == pl.col("re_tick")).alias("same_tick")))
print("cs_pre_restart tick == round_end tick:", tick_chk.group_by("same_tick").len().rows())
# check: round_state round_end row equals cs_pre_restart row (where both exist)
both = post.join(post_re, on=["match_key", "round"], how="inner")
print("rows with both round_state round_end and cs_pre_restart:", both.height,
      "| identical scores:", both.filter((pl.col("t_post") == pl.col("t_post_re")) &
                                        (pl.col("ct_post") == pl.col("ct_post_re"))).height)

J = (RE.unique(subset=["match_key", "round"], keep="first")
       .join(pre, on=["match_key", "round"], how="left")
       .join(post, on=["match_key", "round"], how="left")
       .join(H, on="match_key", how="left"))
print("round_end rounds with pre and post score rows:",
      J.filter(pl.col("t_pre").is_not_null() & pl.col("t_post").is_not_null()).height, "of", J.height)

J = J.with_columns(
    (pl.col("t_post") - pl.col("t_pre")).alias("dt"),
    (pl.col("ct_post") - pl.col("ct_pre")).alias("dc"),
).with_columns(
    pl.when((pl.col("dt") == 1) & (pl.col("dc") == 0)).then(pl.lit(2))
      .when((pl.col("dc") == 1) & (pl.col("dt") == 0)).then(pl.lit(3))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("score_winner"),
    pl.when(pl.col("round") == 13).then(pl.lit("r13"))
      .when(pl.col("round").is_in([25, 28])).then(pl.lit("r25_28"))
      .otherwise(pl.lit("other")).alias("rclass"),
)
print("\n== score increment patterns (dt, dc) over all round_end rounds with score rows")
print(J.filter(pl.col("dt").is_not_null()).group_by(["dt", "dc"]).len().sort(["dt", "dc"]))
print("score_winner null rows:", J.filter(pl.col("score_winner").is_null() & pl.col("dt").is_not_null()).height)

J = J.with_columns((pl.col("score_winner") == pl.col("winner_team_code")).alias("agree"))
K = J.filter(pl.col("score_winner").is_not_null())

print("\n== agreement of round_end winner with scoreboard winner, by build x round class")
print(K.group_by(["build_num", "rclass"]).agg(
    pl.len().alias("n"),
    pl.col("agree").mean().round(4).alias("agree_rate"),
    (~pl.col("agree")).sum().alias("disagree_n")).sort(["build_num", "rclass"]))

print("\n== agreement by build x round index (rounds 1-30), agree_rate (n)")
for b in [9842, 9961]:
    sub = K.filter(pl.col("build_num") == b).group_by("round").agg(
        pl.len().alias("n"), pl.col("agree").mean().alias("rate")).sort("round")
    cells = [f"r{int(r)}:{rt:.2f}({n})" for r, n, rt in sub.select(["round", "n", "rate"]).rows() if r <= 30]
    print(f"build {b}: " + "  ".join(cells))

print("\n== build 9961 round 13 disagreements: round_end winner vs scoreboard winner (pc20 vs other)")
r13 = K.filter((pl.col("build_num") == 9961) & (pl.col("round") == 13))
print(r13.group_by(["player_count", "winner_team_code", "score_winner"]).len().sort(
    ["player_count", "winner_team_code", "score_winner"]))

print("\n== build 9961 round 25 and 28 disagreements")
r2528 = K.filter((pl.col("build_num") == 9961) & pl.col("round").is_in([25, 28]))
print(r2528.group_by(["round", "winner_team_code", "score_winner"]).len().sort(["round", "winner_team_code", "score_winner"]))

# per-match count of score-vs-round_end disagreements in build 9961, round 13 only
per_match = (K.filter((pl.col("build_num") == 9961) & (pl.col("round") == 13))
             .group_by("match_key").agg(pl.len().alias("n_r13"), (~pl.col("agree")).sum().alias("dis_r13")))
print("\nbuild 9961 matches with round 13 score rows:", per_match.height,
      "| matches with >=1 disagreement at r13:", per_match.filter(pl.col("dis_r13") > 0).height,
      "| matches with 0 disagreement:", per_match.filter(pl.col("dis_r13") == 0).height)

# save the joined per-round table for the next steps (scratch, not the project)
SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\D9_9verify")
SCR.mkdir(parents=True, exist_ok=True)
J.write_parquet(SCR / "round_score_join.parquet")
print("\nwrote", SCR / "round_score_join.parquet", "rows", J.height)
