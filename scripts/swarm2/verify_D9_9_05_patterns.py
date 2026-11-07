"""verify_D9_9 step 5: what distinguishes matches by score-stream swap pattern (none / 12 / 8 / 12,27 / 27)?

Per match: rounds present, max round, roster sizes, player_count mode, map, rushb_version, build.
Per pattern x build: bomb-truth agreement and recorded agreement by round index bucket.
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(120)
pl.Config.set_tbl_cols(30)
SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D9_9")
R = pl.read_parquet(SCR / "verify_D9_9_round_oracle.parquet")
M = pl.read_parquet(SCR / "verify_D9_9_match.parquet")

pat = (R.filter(pl.col("swap_sig")).sort(["match_key", "round"])
       .group_by("match_key").agg(pl.col("round").cast(pl.Utf8).str.join(",").alias("swap_pat")))
per = (R.group_by("match_key").agg(pl.col("round").max().alias("max_round"),
                                   pl.col("round").n_unique().alias("n_rounds"),
                                   pl.col("n_T").median().alias("nT_med"),
                                   pl.col("n_CT").median().alias("nCT_med"),
                                   pl.col("player_count").mode().first().alias("pc_mode"),
                                   pl.col("bomb_side").is_not_null().sum().alias("n_bomb_rounds"))
       .join(pat, on="match_key", how="left")
       .with_columns(pl.col("swap_pat").fill_null("none"))
       .join(M.select(["match_key", "build_num", "map_name", "rushb_version"]), on="match_key", how="left"))

print("== per-pattern x build: match counts, max round, rounds, roster medians, player_count mode")
print(per.group_by(["build_num", "swap_pat"]).agg(
    pl.len().alias("matches"),
    pl.col("max_round").median().alias("max_round_med"),
    pl.col("max_round").min().alias("max_round_min"),
    pl.col("max_round").max().alias("max_round_max"),
    pl.col("nT_med").median().alias("nT_med"),
    pl.col("nCT_med").median().alias("nCT_med"),
    pl.col("pc_mode").mode().first().alias("pc_mode"),
    pl.col("n_rounds").mean().round(1).alias("rounds_mean")).sort(["build_num", "swap_pat"]))

print("\n== map_name by pattern (build 9961)")
print(per.filter(pl.col("build_num") == 9961).group_by(["swap_pat", "map_name"]).len().sort(["swap_pat", "len"], descending=[False, True]).head(60))

print("\n== rushb_version by pattern (build 9961)")
print(per.filter(pl.col("build_num") == 9961).group_by(["swap_pat", "rushb_version"]).len().sort(["swap_pat", "len"], descending=[False, True]).head(40))

# bomb truth vs recorded vs score, by pattern and round bucket (bomb rounds only)
bo = R.filter(pl.col("has_outcome") & pl.col("w_score").is_not_null()).join(pat, on="match_key", how="left").with_columns(pl.col("swap_pat").fill_null("none"))
bo = bo.with_columns(pl.when(pl.col("round") <= 8).then(pl.lit("01-08"))
                     .when(pl.col("round") == 9).then(pl.lit("09"))
                     .when(pl.col("round").is_between(10, 12)).then(pl.lit("10-12"))
                     .when(pl.col("round") == 13).then(pl.lit("13"))
                     .when(pl.col("round").is_between(14, 24)).then(pl.lit("14-24"))
                     .otherwise(pl.lit("25+")).alias("rb"))
print("\n== bomb rounds: recorded==bomb truth and score==bomb truth, by build x pattern x round bucket")
print(bo.group_by(["build_num", "swap_pat", "rb"]).agg(
    pl.len().alias("n"),
    (pl.col("winner_team_code") == pl.col("bomb_side")).mean().round(3).alias("rec_eq_bomb"),
    pl.col("score_eq_bomb").mean().round(3).alias("score_eq_bomb")).sort(["build_num", "swap_pat", "rb"]))
