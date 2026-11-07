"""verify_D9_9 step 6: alternative-explanation checks.

(a) the 7 'degenerate roster' bomb rows: identify them (r13, build 9961, recorded == bomb truth), roster, pattern.
(b) label-frame anchor: planter side (bomb_planted player_team_code) and defuser side in bomb rounds by build x round.
(c) label continuity: fraction of players whose team_code flips between consecutive rounds (r-1 -> r), by build x transition.
(d) same-round frame: victim death label vs player_info label of the same player-round, by build x round.
(e) the 2x2 table for 9961 r13 non-bomb: score oracle vs death-check pass pattern (rec only / flip only / both / neither).
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
pat = (R.filter(pl.col("swap_sig")).sort(["match_key", "round"]).group_by("match_key")
       .agg(pl.col("round").cast(pl.Utf8).str.join(",").alias("swap_pat")))
R = R.join(pat, on="match_key", how="left").with_columns(pl.col("swap_pat").fill_null("none"))
PR = pl.read_parquet(SCR / "verify_D9_9_playerround.parquet")
DE = pl.read_parquet(SCR / "verify_D9_9_playerdeath.parquet")
BE = pl.read_parquet(SCR / "verify_D9_9_bombevents.parquet")

print("== (a) bomb rounds in build 9961 round 13: all rows, with w_score null flag and recorded==bomb")
b13 = R.filter((pl.col("build_num") == 9961) & (pl.col("round") == 13) & pl.col("has_outcome"))
print("rows (all bomb rounds r13, 9961):", b13.height,
      "| w_score null:", b13.filter(pl.col("w_score").is_null()).height,
      "| rec==bomb:", b13.filter(pl.col("winner_team_code") == pl.col("bomb_side")).height)
agree = b13.filter(pl.col("winner_team_code") == pl.col("bomb_side"))
print(agree.select(["match_key", "winner_team_code", "bomb_side", "n_T", "n_CT", "n_players_info", "swap_pat",
                    "w_score", "player_count", "T_dead", "CT_dead"]))
print("rosters of all 9961 r13 bomb rows (n_T, n_CT, pattern) counts:")
print(b13.group_by(["n_T", "n_CT", "swap_pat"]).len().sort("len", descending=True).head(20))
print("9961 r13 bomb rows with w_score null by roster:")
print(b13.filter(pl.col("w_score").is_null()).group_by(["n_T", "n_CT"]).len())

print("\n== (b) planter / defuser side in bomb rounds (player frame), by build x round bucket")
pl_side = (BE.filter(pl.col("event_type") == "bomb_planted").group_by(["match_key", "round"])
           .agg(pl.col("player_team_code").drop_nulls().first().alias("planter_side")))
df_side = (BE.filter(pl.col("event_type") == "bomb_defused").group_by(["match_key", "round"])
           .agg(pl.col("player_team_code").drop_nulls().first().alias("defuser_side")))
X = (R.filter(pl.col("has_outcome")).join(pl_side, on=["match_key", "round"], how="left")
     .join(df_side, on=["match_key", "round"], how="left"))
X = X.with_columns(pl.when(pl.col("round") == 13).then(pl.lit("r13"))
                   .when(pl.col("round") == 9).then(pl.lit("r9"))
                   .when(pl.col("round").is_in([25, 28])).then(pl.lit("r25_28"))
                   .otherwise(pl.lit("other")).alias("rb"))
print(X.group_by(["build_num", "rb", "bomb_mismatch"]).agg(
    pl.len().alias("n"),
    (pl.col("planter_side") == 2).sum().alias("planter_T"),
    (pl.col("planter_side") == 3).sum().alias("planter_CT"),
    pl.col("planter_side").is_null().sum().alias("planter_null"),
    (pl.col("defuser_side") == 3).sum().alias("defuser_CT"),
    (pl.col("defuser_side") == 2).sum().alias("defuser_T"),
    pl.col("defuser_side").is_null().sum().alias("defuser_null")).sort(["build_num", "rb", "bomb_mismatch"]))

print("\n== (c) label continuity: fraction of player-rounds whose team_code differs from the previous round (same player, same match)")
PRo = PR.select(["match_key", "round", "player_id", "team_code", "build_num"]).sort(["match_key", "player_id", "round"])
PRo = PRo.with_columns(pl.col("round").shift(1).over(["match_key", "player_id"]).alias("r_prev"),
                       pl.col("team_code").shift(1).over(["match_key", "player_id"]).alias("tc_prev"))
PRo = PRo.filter(pl.col("r_prev") == pl.col("round") - 1)
PRo = PRo.with_columns((pl.col("team_code") != pl.col("tc_prev")).alias("flip"))
trans = PRo.filter(pl.col("round").is_in([8, 9, 10, 12, 13, 14, 24, 25, 26, 27, 28, 29]))
print(trans.group_by(["build_num", "round"]).agg(pl.len().alias("pairs"), pl.col("flip").mean().round(3).alias("frac_flip"))
      .sort(["build_num", "round"]))

print("\n== (d) same-round frame: victim death label vs player_info label (same match, round, player), by build x round")
DEu = DE.select(["match_key", "round", "player_id", "player_team_code", "build_num"]).unique(
    subset=["match_key", "round", "player_id"], keep="first")
J = DEu.join(PR.select(["match_key", "round", "player_id", "team_code"]).unique(
    subset=["match_key", "round", "player_id"], keep="last"), on=["match_key", "round", "player_id"], how="inner")
J = J.with_columns((pl.col("player_team_code") == pl.col("team_code")).alias("agree"))
print(J.filter(pl.col("round").is_in([8, 9, 12, 13, 14, 25, 28])).group_by(["build_num", "round"]).agg(
    pl.len().alias("n"), pl.col("agree").mean().round(4).alias("agree_frac")).sort(["build_num", "round"]))
print("overall agreement:", J.group_by("build_num").agg(pl.col("agree").mean().round(4)).rows())

print("\n== (e) 9961 r13 non-bomb (roster known): score oracle vs death-check pass pattern, by swap pattern")
NB = R.filter((pl.col("build_num") == 9961) & (pl.col("round") == 13) & ~pl.col("has_outcome") & pl.col("n_CT").is_not_null()
              & pl.col("w_score").is_not_null())


def passes(w):
    return ((w == 2) & (pl.col("CT_dead") >= pl.col("n_CT"))) | ((w == 3) & (pl.col("T_dead") >= pl.col("n_T")))


NB = NB.with_columns(passes(pl.col("winner_team_code")).alias("rec_pass"),
                     passes(5 - pl.col("winner_team_code")).alias("flip_pass"),
                     (pl.col("w_score") != pl.col("winner_team_code")).alias("score_says_inverted"))
print(NB.group_by(["swap_pat", "score_says_inverted", "rec_pass", "flip_pass"]).len().sort(["swap_pat", "score_says_inverted", "rec_pass", "flip_pass"]))
