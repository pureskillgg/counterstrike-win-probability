"""verify D1-6 win-reason check (swarm2 verifier).

Independent field: round_end.win_reason_code / win_reason_message (engine's stated reason for the round end).
Question: for the 305 round-9 d12-25 rows where winner_team_code disagrees with the score side, does the stated
win reason agree with winner_team_code or with the score side?

ASSUMPTIONS: same as verify_D1_6_recompute.py (A1 days, A2 player_count == 20, A4 ADJ score side, A8 dup exclusion).
The reason->side mapping is NOT assumed; it is learned from agreeing rows and then applied to disagreeing rows.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(50)

BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
DAYS_HI = [f"{d:02d}" for d in range(12, 26)]


def fl(ch):
    return sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))


def hdr(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


RE = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "winner_team_code", "player_count",
                                            "win_reason_code", "win_reason_message"])
                for f in fl("round_end")], how="vertical_relaxed")
dups = RE.group_by(["match_key", "round"]).agg(pl.len().alias("n")).filter(pl.col("n") > 1)
RE = RE.join(dups.select(["match_key", "round"]), on=["match_key", "round"], how="anti")

RS = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "tick", "t_score", "ct_score"])
                for f in fl("round_state")], how="vertical_relaxed").sort(["match_key", "round", "tick"])
S = RS.group_by(["match_key", "round"], maintain_order=True).agg(
    pl.col("t_score").first().alias("t0"), pl.col("ct_score").first().alias("c0"),
    pl.col("t_score").last().alias("t1"), pl.col("ct_score").last().alias("c1"))
S = S.with_columns(
    pl.when((pl.col("t1") - pl.col("t0")) == 1).then(pl.lit(2))
      .when((pl.col("c1") - pl.col("c0")) == 1).then(pl.lit(3))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("adj_side"))

A = RE.join(S.select(["match_key", "round", "adj_side"]), on=["match_key", "round"], how="left")
A = A.with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"))
A = A.with_columns(pl.when(pl.col("day").is_in(DAYS_HI)).then(pl.lit("d12-25")).otherwise(pl.lit("d01-07")).alias("dg"))
A = A.filter(pl.col("player_count") == 20)
A = A.with_columns(
    pl.when(pl.col("adj_side").is_null()).then(pl.lit("no_side"))
      .when(pl.col("adj_side") == pl.col("winner_team_code")).then(pl.lit("agree"))
      .otherwise(pl.lit("disagree")).alias("status"))

# ----------------------------------------------------------------------------- reason vocabulary
hdr("W0 win_reason vocabulary (player_count == 20, all rounds): code, message, winner distribution")
print(A.group_by(["win_reason_code", "win_reason_message"]).agg(
    pl.len().alias("n"),
    (pl.col("winner_team_code") == 2).sum().alias("winner_T"),
    (pl.col("winner_team_code") == 3).sum().alias("winner_CT")).sort("n", descending=True))

# ----------------------------------------------------------------------------- learn mapping on agreeing rows
hdr("W1 reason vs winner on AGREEING rows (score side == winner), all rounds, d12-25 and d01-07")
ag = A.filter(pl.col("status") == "agree")
print(ag.group_by(["win_reason_message", "winner_team_code"]).len().sort("len", descending=True).head(40))

# ----------------------------------------------------------------------------- headline: round 9 disagreeing rows
hdr("W2 round 9, d12-25: reason message x winner x score side, by status")
x9 = A.filter((pl.col("round") == 9) & (pl.col("dg") == "d12-25") & (pl.col("status") != "no_side"))
print(x9.group_by(["status", "win_reason_message", "winner_team_code", "adj_side"]).len()
      .sort(["status", "len"], descending=[False, True]))

hdr("W3 round 9, d12-25, DISAGREE rows only: win reason message counts")
dis9 = x9.filter(pl.col("status") == "disagree")
print(dis9.group_by("win_reason_message").len().sort("len", descending=True))

hdr("W4 round 9, d12-25, AGREE rows only: win reason message counts (comparison)")
agr9 = x9.filter(pl.col("status") == "agree")
print(agr9.group_by("win_reason_message").len().sort("len", descending=True))

hdr("W5 round 9 d12-25 disagree rows: win reason implied side vs winner vs score")
# implied side: the side that the reason message names as the winner, learned from agreeing rows
# (for each reason, the modal winner_team_code on agreeing rows across ALL rounds)
modal = (ag.group_by(["win_reason_message", "winner_team_code"]).len()
         .sort(["win_reason_message", "len"], descending=[False, True])
         .group_by("win_reason_message", maintain_order=True).first()
         .select(["win_reason_message", pl.col("winner_team_code").alias("reason_side")]))
purity = (ag.group_by(["win_reason_message", "winner_team_code"]).len()
          .group_by("win_reason_message").agg(pl.col("len").max().alias("mx"), pl.col("len").sum().alias("tot"))
          .with_columns((pl.col("mx") / pl.col("tot")).alias("purity")))
print("reason->side purity on agreeing rows (all rounds):")
print(purity.sort("tot", descending=True))
d9m = dis9.join(modal, on="win_reason_message", how="left")
print("disagree round 9 rows: reason_side == winner_team_code:", d9m.filter(pl.col("reason_side") == pl.col("winner_team_code")).height,
      " reason_side == adj_side:", d9m.filter(pl.col("reason_side") == pl.col("adj_side")).height,
      " reason unmapped:", d9m.filter(pl.col("reason_side").is_null()).height, " total:", d9m.height)

hdr("W6 same test on the agreeing round-9 rows (sanity: reason should match winner ~always)")
agm = agr9.join(modal, on="win_reason_message", how="left")
print("agree round 9 rows: reason_side == winner:", agm.filter(pl.col("reason_side") == pl.col("winner_team_code")).height,
      " of", agm.height)

hdr("W7 all rounds d12-25 disagree rows by round: reason_side == winner vs == score")
dall = A.filter((pl.col("status") == "disagree") & (pl.col("dg") == "d12-25")).join(modal, on="win_reason_message", how="left")
print(dall.group_by("round").agg(
    pl.len().alias("n_disagree"),
    (pl.col("reason_side") == pl.col("winner_team_code")).sum().alias("reason_matches_winner"),
    (pl.col("reason_side") == pl.col("adj_side")).sum().alias("reason_matches_score"),
    pl.col("reason_side").is_null().sum().alias("unmapped")).sort("round"))
