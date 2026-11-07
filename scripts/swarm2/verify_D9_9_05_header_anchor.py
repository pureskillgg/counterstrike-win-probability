"""verify D9-9, step 5: match-level anchor from header final scores (2024-02 tome, read-only).

For each match the starters' final scores in the header (t_starters_score_final = rounds won by the
team that started on T; ct_starters_score_final = rounds won by the team that started on CT) are
compared with two round-winner sequences:
  (a) round_end.winner_team_code as recorded (RE)
  (b) scoreboard winner from round_state score increments (SB), see verify_D9_9_03

Starters' side per round: starters_T is on T (code 2) in round 1. Each side swap is detected from the
scoreboard: pre(r) equal to (ct_post(r-1), t_post(r-1)) means a swap at the boundary r-1 -> r; pre(r)
equal to (t_post(r-1), ct_post(r-1)) means no swap. Rounds where neither holds are counted as ambiguous.

ASSUMPTIONS (ASSUMPTION):
  - header final scores are the game's own match totals and are independent of round_end winners
    (not verified against an external source).
  - the number of rounds in the header total equals the number of round_end rows for the match.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(24)
SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\D9_9verify")
J = pl.read_parquet(SCR / "round_score_join.parquet")
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
H = pl.scan_parquet(sorted(glob.glob(str(BASE / "header" / "dataframe_*")))).select(
    ["key", "t_starters_score_final", "ct_starters_score_final", "build_num"]).collect().rename(
    {"key": "match_key"})

# J has one row per (match, round) with pre (t_pre, ct_pre) and post (t_post, ct_post) where available.
J = J.sort(["match_key", "round"])
J = J.with_columns(
    pl.col("t_post").shift(1).over("match_key").alias("prev_t_post"),
    pl.col("ct_post").shift(1).over("match_key").alias("prev_ct_post"),
    pl.col("round").shift(1).over("match_key").alias("prev_round"),
)
# boundary status between round r-1 and r
J = J.with_columns(
    pl.when(pl.col("prev_round").is_null()).then(pl.lit("first"))
      .when((pl.col("prev_round") == pl.col("round") - 1) &
            (pl.col("t_pre") == pl.col("prev_t_post")) & (pl.col("ct_pre") == pl.col("prev_ct_post")))
      .then(pl.lit("noswap"))
      .when((pl.col("prev_round") == pl.col("round") - 1) &
            (pl.col("t_pre") == pl.col("prev_ct_post")) & (pl.col("ct_pre") == pl.col("prev_t_post")))
      .then(pl.lit("swap"))
      .when(pl.col("prev_round") == pl.col("round") - 1).then(pl.lit("ambig"))
      .otherwise(pl.lit("gap")).alias("boundary"),
)
print("boundary status counts (all rounds with pre score):")
print(J.filter(pl.col("t_pre").is_not_null()).group_by("boundary").len().sort("boundary"))

# swap parity per match, and starters' side per round
J = J.with_columns(
    (pl.col("boundary") == "swap").cast(pl.Int64).alias("is_swap"),
).with_columns(
    pl.col("is_swap").cum_sum().over("match_key").alias("swap_cum"),
)
J = J.with_columns(
    pl.when((pl.col("swap_cum") % 2) == 0).then(pl.lit(2)).otherwise(pl.lit(3)).alias("starters_T_side"),
)
# per-round starter win indicator under each winner series
J = J.with_columns(
    (pl.col("starters_T_side") == pl.col("winner_team_code")).cast(pl.Int64).alias("stT_win_re"),
    (pl.col("starters_T_side") == pl.col("score_winner")).cast(pl.Int64).alias("stT_win_sb"),
)
# keep only rounds with score rows and contiguous boundaries (no gaps / ambiguity) per match
ok_match = (J.group_by("match_key").agg(
    pl.len().alias("n"),
    (pl.col("score_winner").is_null()).sum().alias("n_sb_null"),
    (pl.col("boundary") == "ambig").sum().alias("n_ambig"),
    (pl.col("boundary") == "gap").sum().alias("n_gap"),
    pl.col("round").max().alias("max_round"),
).with_columns(
    ((pl.col("n_sb_null") == 0) & (pl.col("n_ambig") == 0) & (pl.col("n_gap") == 0)).alias("clean")
))
print("\nmatches:", ok_match.height, "| clean (no null/ambig/gap):", ok_match.filter(pl.col("clean")).height)

tot = (J.group_by("match_key").agg(
    pl.col("stT_win_re").sum().alias("stT_re"),
    pl.col("stT_win_sb").sum().alias("stT_sb"),
    pl.len().alias("n_rounds"),
    (pl.col("boundary") == "swap").sum().alias("n_swaps"),
    pl.col("round").filter(pl.col("boundary") == "swap").min().alias("first_swap_round"),
    pl.col("round").filter(pl.col("boundary") == "swap").max().alias("last_swap_round"),
    pl.col("round").filter(pl.col("boundary") == "swap").implode().alias("swap_rounds"),
).join(ok_match.select(["match_key", "clean"]), on="match_key")
    .join(H, on="match_key", how="left"))
tot = tot.with_columns(
    (pl.col("stT_re") == pl.col("t_starters_score_final")).alias("re_matches_header_T"),
    (pl.col("stT_sb") == pl.col("t_starters_score_final")).alias("sb_matches_header_T"),
    ((pl.col("n_rounds") - pl.col("stT_re")) == pl.col("ct_starters_score_final")).alias("re_matches_header_CT"),
    ((pl.col("n_rounds") - pl.col("stT_sb")) == pl.col("ct_starters_score_final")).alias("sb_matches_header_CT"),
)
C = tot.filter(pl.col("clean"))
print("\n== clean matches: header agreement (T-starter total) by build")
print(C.group_by("build_num").agg(
    pl.len().alias("n"),
    pl.col("re_matches_header_T").mean().round(4).alias("re_T"),
    pl.col("sb_matches_header_T").mean().round(4).alias("sb_T"),
    pl.col("re_matches_header_CT").mean().round(4).alias("re_CT"),
    pl.col("sb_matches_header_CT").mean().round(4).alias("sb_CT"),
).sort("build_num"))

print("\n== clean matches: header agreement by build and whether the match had a swap at round 13")
C = C.with_columns(
    pl.col("swap_rounds").list.contains(13).alias("has_r13_swap"),
)
print(C.group_by(["build_num", "has_r13_swap"]).agg(
    pl.len().alias("n"),
    pl.col("re_matches_header_T").mean().round(4).alias("re_T"),
    pl.col("sb_matches_header_T").mean().round(4).alias("sb_T"),
    (pl.col("re_matches_header_T") & pl.col("re_matches_header_CT")).mean().round(4).alias("re_both"),
    (pl.col("sb_matches_header_T") & pl.col("sb_matches_header_CT")).mean().round(4).alias("sb_both"),
).sort(["build_num", "has_r13_swap"]))

print("\n== swap rounds observed (clean matches, build 9961): count of matches by swap-round pattern")
w = C.filter(pl.col("build_num") == 9961).group_by(pl.col("swap_rounds").list.eval(pl.element().cast(pl.Utf8)).list.join(",")).len().sort("len", descending=True)
print(w.head(12))

print("\n== build 9961 clean matches where RE and SB disagree on the starter total (count):")
d = C.filter((pl.col("build_num") == 9961) & (pl.col("stT_re") != pl.col("stT_sb")))
print("n =", d.height, "of", C.filter(pl.col("build_num") == 9961).height)
print(d.select(["match_key", "stT_re", "stT_sb", "t_starters_score_final", "ct_starters_score_final",
                "n_rounds", "swap_rounds", "re_matches_header_T", "sb_matches_header_T"]).head(10))

SCR.mkdir(parents=True, exist_ok=True)
tot.write_parquet(SCR / "match_header_anchor.parquet")
print("\nwrote", SCR / "match_header_anchor.parquet", tot.height)
