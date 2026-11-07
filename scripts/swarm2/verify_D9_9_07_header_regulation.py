"""verify D9-9, step 7: header final-score anchor on regulation matches (2024-02 tome, read-only).

Regulation matches only (max round <= 24, so no overtime side schedule is needed). Standard
side schedule: the starters' team (starters_T) is on T in rounds 1-12 and on CT in rounds 13-24.
For each match, the starters' total implied by a winner series is compared with the header's
t_starters_score_final (starters_T total) and ct_starters_score_final (starters_CT total).

Winner series:
  RE: round_end.winner_team_code as recorded
  SB: scoreboard winner (round_state score increments; post-round score = cs_pre_restart, and for the
      final round the cs_win_panel_match row, which is the final score in regulation)

ASSUMPTIONS (ASSUMPTION):
  - regulation matches end at 13 wins, so the final round is the 13th win for one side.
  - header t/ct starters' finals are the game's own totals (not verified externally).
  - the standard halftime swap after round 12 (no swap at other boundaries in rounds 1-24).
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(24)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\D9_9verify")

J = pl.read_parquet(SCR / "round_score_join.parquet")  # per (match, round): winner_team_code, pre, post
RS = pl.scan_parquet(sorted(glob.glob(str(BASE / f"basic_round_state{SUF}" / "dataframe_*")))).select(
    ["match_key", "round", "tick", "event_type", "t_score", "ct_score"]).collect()
H = pl.scan_parquet(sorted(glob.glob(str(BASE / "header" / "dataframe_*")))).select(
    ["key", "build_num", "t_starters_score_final", "ct_starters_score_final"]).collect().rename({"key": "match_key"})

cwp = RS.filter(pl.col("event_type") == "cs_win_panel_match").select(
    ["match_key", pl.col("t_score").alias("t_fin"), pl.col("ct_score").alias("ct_fin")]).unique(
    subset=["match_key"], keep="last")

rounds = J.group_by("match_key").agg(
    pl.col("round").max().alias("max_round"),
    pl.len().alias("n_rows"),
    pl.col("round").n_unique().alias("n_unique"),
)
reg = rounds.filter(pl.col("max_round") <= 24).select("match_key")
print("matches:", rounds.height, "| regulation (max round <= 24):", reg.height)
contig = rounds.filter((pl.col("max_round") == pl.col("n_unique")) & (pl.col("max_round") <= 24))
print("regulation and contiguous 1..N:", contig.height)

K = J.join(contig.select("match_key"), on="match_key", how="inner").join(cwp, on="match_key", how="left")
last_r = K.group_by("match_key").agg(pl.col("round").max().alias("lr"))
K = K.join(last_r, on="match_key", how="left")
K = K.with_columns(
    pl.when(pl.col("round") == pl.col("lr")).then(pl.col("t_fin")).otherwise(pl.col("t_post")).alias("t_post2"),
    pl.when(pl.col("round") == pl.col("lr")).then(pl.col("ct_fin")).otherwise(pl.col("ct_post")).alias("ct_post2"),
)
K = K.with_columns(
    (pl.col("t_post2") - pl.col("t_pre")).alias("dt"),
    (pl.col("ct_post2") - pl.col("ct_pre")).alias("dc"),
).with_columns(
    pl.when((pl.col("dt") == 1) & (pl.col("dc") == 0)).then(pl.lit(2))
      .when((pl.col("dc") == 1) & (pl.col("dt") == 0)).then(pl.lit(3))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("sb"),
    pl.when(pl.col("round") <= 12).then(pl.lit(2)).otherwise(pl.lit(3)).alias("st_side"),
)
print("\nrows:", K.height, "| SB null:", K.filter(pl.col("sb").is_null()).height,
      "| final-round rows with cs_win_panel_match:", K.filter((pl.col("round") == pl.col("lr")) & pl.col("t_fin").is_not_null()).height,
      "of", K.filter(pl.col("round") == pl.col("lr")).height)

# per-match totals
T = (K.group_by("match_key").agg(
    pl.len().alias("n_rounds"),
    (pl.col("sb").is_null()).sum().alias("sb_null"),
    (pl.col("winner_team_code") == pl.col("st_side")).sum().alias("stT_re"),
    (pl.col("sb") == pl.col("st_side")).sum().alias("stT_sb"),
    (pl.col("winner_team_code") != pl.col("sb")).sum().alias("n_disagree"),
)
     .join(H, on="match_key", how="left"))
T = T.filter(pl.col("sb_null") == 0)
T = T.with_columns(
    (pl.col("n_rounds") - pl.col("stT_re")).alias("ctT_re"),
    (pl.col("n_rounds") - pl.col("stT_sb")).alias("ctT_sb"),
    (pl.col("stT_re") == pl.col("t_starters_score_final")).alias("re_T_ok"),
    (pl.col("stT_sb") == pl.col("t_starters_score_final")).alias("sb_T_ok"),
    (pl.col("ctT_re") == pl.col("ct_starters_score_final")).alias("re_CT_ok"),
    (pl.col("ctT_sb") == pl.col("ct_starters_score_final")).alias("sb_CT_ok"),
)
T = T.with_columns(
    (pl.col("re_T_ok") & pl.col("re_CT_ok")).alias("re_both"),
    (pl.col("sb_T_ok") & pl.col("sb_CT_ok")).alias("sb_both"),
    (pl.col("n_disagree") > 0).alias("re_sb_disagree"),
)
print("\n== regulation matches with complete SB: header agreement (both starters' totals) by build")
print(T.group_by("build_num").agg(
    pl.len().alias("n"),
    pl.col("re_both").mean().round(4).alias("RE_both_match"),
    pl.col("sb_both").mean().round(4).alias("SB_both_match"),
    pl.col("re_sb_disagree").sum().alias("matches_with_RE_SB_disagree"),
).sort("build_num"))

print("\n== build 9961: header agreement split by whether RE and SB disagree anywhere in the match")
print(T.filter(pl.col("build_num") == 9961).group_by("re_sb_disagree").agg(
    pl.len().alias("n"),
    pl.col("re_both").mean().round(4).alias("RE_both"),
    pl.col("sb_both").mean().round(4).alias("SB_both"),
).sort("re_sb_disagree"))

print("\n== build 9842: header agreement split by RE/SB disagreement (should be none)")
print(T.filter(pl.col("build_num") == 9842).group_by("re_sb_disagree").agg(
    pl.len().alias("n"),
    pl.col("re_both").mean().round(4).alias("RE_both"),
    pl.col("sb_both").mean().round(4).alias("SB_both"),
).sort("re_sb_disagree"))

print("\n== build 9961 matches where RE and SB disagree: sample")
print(T.filter((pl.col("build_num") == 9961) & pl.col("re_sb_disagree")).select(
    ["match_key", "n_rounds", "stT_re", "stT_sb", "t_starters_score_final", "ct_starters_score_final",
     "re_both", "sb_both", "n_disagree"]).head(8))

T.write_parquet(SCR / "header_regulation_check.parquet")
print("\nwrote", SCR / "header_regulation_check.parquet", T.height)
