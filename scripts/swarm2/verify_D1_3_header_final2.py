"""verify D1-3: team-level match arbitration with the demo header final scores (read-only, 2024 tome).

Changes vs verify_D1_3_header_final.py:
  - team A = the 5 players on T in the FIRST round with player_info for the match (works for d01-07 too,
    where round 1 has no player_info rows).
  - S2 is computed at TEAM level across side swaps: team A's score before round r is read from the field
    of A's side at round r; its score after round r is read from the field of A's side at round r+1
    (pre-values of round r+1, which already reflect the halftime / OT swap). Gain of +1 for one team
    gives the winner side. Rounds with no next round use round_officially_ended as post.
  - Matches are split by whether every round has a defined S2 team winner.

ASSUMPTIONS:
  A1 header t_starters_score_final = final score of team A (the team on T in the first round with data).
  A2 team score fields are team-level and swap with sides at halftime / OT (verified indirectly by the
     round-12 trace: pre/post fields cross the swap, and round 13 starts where round 12 ended).
  A3 sideA(r) = majority side of team A's present players in round r (player_info team codes).
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(20)
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
DAYS_HI = [f"{d:02d}" for d in range(12, 26)]


def rd(ch, cols):
    fs = sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


def hdr(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


HD = pl.concat([pl.read_parquet(f, columns=["key", "t_starters_score_final", "ct_starters_score_final"])
                for f in sorted(glob.glob(str(BASE / "header" / "dataframe_*")))], how="vertical_relaxed")
HD = HD.unique("key", keep="first").rename({"key": "match_key", "t_starters_score_final": "hA", "ct_starters_score_final": "hB"})

PI = rd("player_info", ["match_key", "round", "player_id_fixed", "team_code"]).unique(["match_key", "round", "player_id_fixed"], keep="first")
RE = rd("round_end", ["match_key", "round", "winner_team_code"]).unique(["match_key", "round"], keep="first")
RS = rd("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"]).sort(["match_key", "round", "tick"])
BS = rd("bomb_state", ["match_key", "round", "event_type"])

# team A: T players of the first round with player_info (5 T players)
first_r = PI.group_by("match_key").agg(pl.col("round").min().alias("r0"))
TA = (PI.join(first_r, on="match_key").filter((pl.col("round") == pl.col("r0")) & (pl.col("team_code") == 2))
      .select(["match_key", "player_id_fixed"]).unique().with_columns(pl.lit(True).alias("isA")))
PIA = PI.join(TA, on=["match_key", "player_id_fixed"], how="left").with_columns(pl.col("isA").fill_null(False))
sideA = (PIA.filter(pl.col("isA")).group_by(["match_key", "round"])
         .agg((pl.col("team_code") == 2).sum().alias("aT"), (pl.col("team_code") == 3).sum().alias("aCT"))
         .with_columns(pl.when(pl.col("aT") > pl.col("aCT")).then(2).when(pl.col("aCT") > pl.col("aT")).then(3).otherwise(None).alias("sideA"))
         .select(["match_key", "round", "sideA"]))

# score pre-values per round (first row, or round_freeze_end first row), and post for the last round
pre = (RS.filter(pl.col("event_type") == "round_freeze_end").group_by(["match_key", "round"])
       .agg(pl.col("t_score").first().alias("tp"), pl.col("ct_score").first().alias("cp")))
pre0 = RS.group_by(["match_key", "round"]).agg(pl.col("t_score").first().alias("tp0"), pl.col("ct_score").first().alias("cp0"))
pv = pre0.join(pre, on=["match_key", "round"], how="left").with_columns(
    pl.coalesce(["tp", "tp0"]).alias("tpre"), pl.coalesce(["cp", "cp0"]).alias("cpre")).select(["match_key", "round", "tpre", "cpre"])
post_last = (RS.filter(pl.col("event_type") == "round_officially_ended").group_by(["match_key", "round"])
             .agg(pl.col("t_score").last().alias("tq"), pl.col("ct_score").last().alias("cq")))
post_last0 = RS.group_by(["match_key", "round"]).agg(pl.col("t_score").last().alias("tq0"), pl.col("ct_score").last().alias("cq0"))
pl_ = post_last0.join(post_last, on=["match_key", "round"], how="left").with_columns(
    pl.coalesce(["tq", "tq0"]).alias("tqf"), pl.coalesce(["cq", "cq0"]).alias("cqf")).select(["match_key", "round", "tqf", "cqf"])

# round table with next-round pre values
R = (RE.join(sideA, on=["match_key", "round"], how="left").join(pv, on=["match_key", "round"], how="left")
     .join(pl_, on=["match_key", "round"], how="left"))
nxt = pv.with_columns((pl.col("round") - 1).alias("round")).rename({"tpre": "tnext", "cpre": "cnext"})
R = R.join(nxt, on=["match_key", "round"], how="left").join(sideA.with_columns((pl.col("round") - 1).alias("round")).rename({"sideA": "sideA_next"}),
                                                          on=["match_key", "round"], how="left")
# A pre and post
R = R.with_columns(
    pl.when(pl.col("sideA") == 2).then(pl.col("tpre")).otherwise(pl.col("cpre")).alias("A_pre"),
    pl.when(pl.col("sideA") == 2).then(pl.col("cpre")).otherwise(pl.col("tpre")).alias("B_pre"),
)
# post: use next-round pre values with the next round's side of A; last round uses round_officially_ended fields (side of A this round)
R = R.with_columns(
    pl.when(pl.col("tnext").is_not_null()).then(
        pl.when(pl.col("sideA_next") == 2).then(pl.col("tnext")).otherwise(pl.col("cnext"))
    ).otherwise(
        pl.when(pl.col("sideA") == 2).then(pl.col("tqf")).otherwise(pl.col("cqf"))
    ).alias("A_post"),
    pl.when(pl.col("tnext").is_not_null()).then(
        pl.when(pl.col("sideA_next") == 2).then(pl.col("cnext")).otherwise(pl.col("tnext"))
    ).otherwise(
        pl.when(pl.col("sideA") == 2).then(pl.col("cqf")).otherwise(pl.col("tqf"))
    ).alias("B_post"))
R = R.with_columns((pl.col("A_post") - pl.col("A_pre")).alias("gA"), (pl.col("B_post") - pl.col("B_pre")).alias("gB"))
R = R.with_columns(
    pl.when((pl.col("gA") == 1) & (pl.col("gB") == 0)).then(pl.col("sideA"))
      .when((pl.col("gB") == 1) & (pl.col("gA") == 0)).then(5 - pl.col("sideA"))
      .otherwise(None).alias("S2T"))
# bomb side and kill side kept out here; only WIN and S2T are arbitrated
R = R.with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day")).with_columns(
    pl.when(pl.col("day").is_in(DAYS_HI)).then(pl.lit("d12-25")).otherwise(pl.lit("d01-07")).alias("dg"))
R = R.filter(pl.col("sideA").is_not_null())
R = R.with_columns(pl.col("winner_team_code").alias("WIN"))

hdr("0. coverage and S2T validity")
print("rounds with sideA:", R.height, "  S2T defined:", R.filter(pl.col("S2T").is_not_null()).height)
print("S2T defined by round number (d12-25), first 16:")
print(R.filter(pl.col("dg") == "d12-25").group_by("round").agg(pl.len().alias("n"), pl.col("S2T").is_not_null().sum().alias("S2T_def"))
      .sort("round").head(16))
print("S2T vs WIN agreement where both defined, by dg:")
both = R.filter(pl.col("S2T").is_not_null())
print(both.group_by("dg").agg(pl.len().alias("n"), (pl.col("S2T") != pl.col("WIN")).sum().alias("S2T_ne_WIN")))
print("S2T vs WIN disagreement by round number (d12-25):")
print(both.filter((pl.col("dg") == "d12-25") & (pl.col("S2T") != pl.col("WIN"))).group_by("round").len().sort("round"))
print("S2T vs WIN disagreement by round number (d01-07):")
print(both.filter((pl.col("dg") == "d01-07") & (pl.col("S2T") != pl.col("WIN"))).group_by("round").len().sort("round"))

# per-match totals
M = (R.group_by("match_key").agg(
        pl.col("dg").first().alias("dg"),
        pl.len().alias("n_rounds"),
        (pl.col("WIN") == pl.col("sideA")).sum().alias("A_WIN"),
        (pl.col("WIN") == 5 - pl.col("sideA")).sum().alias("B_WIN"),
        pl.col("S2T").is_not_null().sum().alias("n_S2T"),
        (pl.col("S2T") == pl.col("sideA")).sum().alias("A_S2T"),
        (pl.col("S2T") == 5 - pl.col("sideA")).sum().alias("B_S2T"),
        (pl.col("S2T") != pl.col("WIN")).sum().alias("n_disagree"))
     .join(HD, on="match_key", how="left"))
M = M.with_columns(
    ((pl.col("A_WIN") == pl.col("hA")) & (pl.col("B_WIN") == pl.col("hB"))).alias("ok_WIN"),
    ((pl.col("A_S2T") == pl.col("hA")) & (pl.col("B_S2T") == pl.col("hB"))).alias("ok_S2T"),
    (pl.col("n_S2T") == pl.col("n_rounds")).alias("S2T_complete"))

hdr("1. match-level arbitration: header final scores vs WIN totals and S2T totals (matches with a header)")
print("matches with sideA:", M.height, " with header:", M.filter(pl.col("hA").is_not_null()).height)
print(M.group_by("dg").agg(pl.len().alias("n"), pl.col("ok_WIN").sum().alias("ok_WIN"), pl.col("ok_S2T").sum().alias("ok_S2T"),
                           pl.col("S2T_complete").sum().alias("S2T_complete")).sort("dg"))
print("\nmatches where S2T is complete (every round defined):")
C = M.filter(pl.col("S2T_complete"))
print(C.group_by(["dg", "ok_WIN", "ok_S2T"]).len().sort(["dg", "ok_WIN", "ok_S2T"]))
print("\nd12-25 complete matches where WIN fails: count", C.filter((pl.col("dg") == "d12-25") & ~pl.col("ok_WIN")).height,
      " of which S2T holds:", C.filter((pl.col("dg") == "d12-25") & ~pl.col("ok_WIN") & pl.col("ok_S2T")).height)
print("d12-25 complete matches where S2T fails: count", C.filter((pl.col("dg") == "d12-25") & ~pl.col("ok_S2T")).height)
print("\nd12-25 match-level: number of matches with n_disagree >= 1 (WIN vs S2T on defined rounds):",
      M.filter((pl.col("dg") == "d12-25") & (pl.col("n_disagree") >= 1)).height, " of", M.filter(pl.col("dg") == "d12-25").height)
print("d01-07 match-level: number of matches with n_disagree >= 1:",
      M.filter((pl.col("dg") == "d01-07") & (pl.col("n_disagree") >= 1)).height, " of", M.filter(pl.col("dg") == "d01-07").height)
print("\nexamples of d12-25 complete matches where WIN fails (A_WIN, B_WIN, A_S2T, B_S2T, hA, hB, n_disagree):")
print(C.filter((pl.col("dg") == "d12-25") & ~pl.col("ok_WIN")).select(["match_key", "A_WIN", "B_WIN", "A_S2T", "B_S2T", "hA", "hB", "n_disagree"]).head(8))

hdr("2. orientation test: does the header match team A/B swapped? (hA<->hB)")
M = M.with_columns(
    ((pl.col("A_WIN") == pl.col("hB")) & (pl.col("B_WIN") == pl.col("hA"))).alias("ok_WIN_sw"),
    ((pl.col("A_S2T") == pl.col("hB")) & (pl.col("B_S2T") == pl.col("hA"))).alias("ok_S2T_sw"),
    ((pl.col("hA") + pl.col("hB")) == pl.col("n_S2T")).alias("hsum_eq_nS2T"),
    ((pl.col("hA") + pl.col("hB")) == pl.col("n_rounds")).alias("hsum_eq_nrounds"))
print(M.group_by("dg").agg(pl.len().alias("n"),
                           pl.col("ok_WIN").sum().alias("ok_WIN"), pl.col("ok_WIN_sw").sum().alias("ok_WIN_swapped"),
                           pl.col("ok_S2T").sum().alias("ok_S2T"), pl.col("ok_S2T_sw").sum().alias("ok_S2T_swapped"),
                           pl.col("hsum_eq_nS2T").sum().alias("hsum_eq_nS2T_defined"),
                           pl.col("hsum_eq_nrounds").sum().alias("hsum_eq_nrounds")).sort("dg"))
print("\nd01-07 sample (A_WIN, B_WIN, A_S2T, B_S2T, hA, hB, n_rounds, n_S2T):")
print(M.filter(pl.col("dg") == "d01-07").select(["match_key", "A_WIN", "B_WIN", "A_S2T", "B_S2T", "hA", "hB", "n_rounds", "n_S2T"]).head(6))
print("\nd12-25 complete matches, orientation-swapped S2T holds:", C.height and M.filter(pl.col("S2T_complete") & (pl.col("dg") == "d12-25") & pl.col("ok_S2T_sw")).height,
      " ; orientation-as-coded S2T holds:", M.filter(pl.col("S2T_complete") & (pl.col("dg") == "d12-25") & pl.col("ok_S2T")).height)
