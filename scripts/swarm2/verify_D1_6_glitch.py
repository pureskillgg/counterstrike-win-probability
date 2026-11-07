"""verify D1-6 glitch test (swarm2 verifier).

Q1 first divergence: for each match (d12-25 and d01-07, player_count == 20, rounds 1..13 all present),
   find the first round r where the in-game score at the first round_state row of round r does NOT equal the
   winner-based cumulative totals through round r-1 (T = count of winner_team_code 2, CT = count of 3).
   Reported for the 305 set and for the d12-25 control (matches not in the 305 set).
Q2 round-8 signature: within-round change of ADJ score at round 8 in the 305 set vs control, and whether the
   round-8 end state equals the swapped winner totals (T and CT interchanged).
Q3 overlap: 305 set vs (a) matches with a round-13 ADJ disagreement, (b) matches with a round-8 null side.
Q4 bomb-defused rows by reason, all rounds d12-25 (pc20): is "defused + Terrorists_Win" unique to round 9?

ASSUMPTIONS: A1 days; A2 player_count == 20 for the 305 set and for the control; A8 duplicate exclusion;
A10 ASSUMPTION: a divergence in cumulative totals is attributed to the first round whose start state is off.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
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
                                            "win_reason_message"])
                for f in fl("round_end")], how="vertical_relaxed")
dups = RE.group_by(["match_key", "round"]).agg(pl.len().alias("n")).filter(pl.col("n") > 1)
RE = RE.join(dups.select(["match_key", "round"]), on=["match_key", "round"], how="anti")
RE = RE.with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"))
RE = RE.with_columns(pl.when(pl.col("day").is_in(DAYS_HI)).then(pl.lit("d12-25")).otherwise(pl.lit("d01-07")).alias("dg"))

RS = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "tick", "t_score", "ct_score"])
                for f in fl("round_state")], how="vertical_relaxed").sort(["match_key", "round", "tick"])
S = RS.group_by(["match_key", "round"], maintain_order=True).agg(
    pl.col("t_score").first().alias("t0"), pl.col("ct_score").first().alias("c0"),
    pl.col("t_score").last().alias("t1"), pl.col("ct_score").last().alias("c1"))
S = S.with_columns(
    pl.when((pl.col("t1") - pl.col("t0")) == 1).then(pl.lit(2))
      .when((pl.col("c1") - pl.col("c0")) == 1).then(pl.lit(3))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("adj_side"))

M = RE.join(S, on=["match_key", "round"], how="left")
M = M.with_columns((pl.col("adj_side").is_not_null() & (pl.col("adj_side") != pl.col("winner_team_code"))).alias("adj_mm"))

# 305 set and control (player_count == 20, d12-25)
set305 = set(M.filter((pl.col("round") == 9) & (pl.col("dg") == "d12-25") & (pl.col("player_count") == 20)
                      & pl.col("adj_mm"))["match_key"].to_list())
mk_pc20 = M.filter((pl.col("dg") == "d12-25") & (pl.col("player_count") == 20))["match_key"].unique().to_list()
control = [m for m in mk_pc20 if m not in set305]
print("305 set:", len(set305), " d12-25 pc20 control matches (not in 305):", len(control))

# ----------------------------------------------------------------------------- Q1 first divergence
# per match: winner-based cumulative totals and in-game start state for rounds 1..13
W = M.filter(pl.col("round") <= 13).sort(["match_key", "round"]).with_columns(
    (pl.col("winner_team_code") == 2).cast(pl.Int64).cum_sum().over("match_key").alias("T_cum"),
    (pl.col("winner_team_code") == 3).cast(pl.Int64).cum_sum().over("match_key").alias("CT_cum"))
# totals through round r-1, aligned to round r
W = W.with_columns(
    pl.col("T_cum").shift(1).over("match_key").alias("T_prev"),
    pl.col("CT_cum").shift(1).over("match_key").alias("CT_prev"),
    pl.col("round").shift(1).over("match_key").alias("round_prev"))
W = W.filter(pl.col("round_prev") == pl.col("round") - 1)
# in-game start state of round r (first round_state row)
W = W.join(S.select(["match_key", "round", "t0", "c0"]), on=["match_key", "round"], how="left")
W = W.with_columns(
    ((pl.col("t0") == pl.col("T_prev")) & (pl.col("c0") == pl.col("CT_prev"))).alias("ok_direct"),
    ((pl.col("t0") == pl.col("CT_prev")) & (pl.col("c0") == pl.col("T_prev"))).alias("ok_swapped"))
W = W.with_columns(((pl.col("ok_direct")) | (pl.col("ok_swapped"))).alias("ok_any"))


def first_div(mks, label):
    x = W.filter(pl.col("match_key").is_in(mks) & (pl.col("player_count") == 20))
    rows = []
    for mk, g in x.group_by("match_key", maintain_order=True):
        g = g.sort("round")
        bad = g.filter(~pl.col("ok_direct"))
        first_bad = int(bad["round"].min()) if bad.height else None
        rows.append({"match_key": mk[0], "first_bad_round": first_bad,
                     "first_bad_is_swapped": (bool(bad.filter(pl.col("ok_swapped"))["round"].min() is not None)
                                              if bad.height else None)})
    d = pl.DataFrame(rows)
    print(f"\n{label}: matches={d.height}")
    print(d.group_by("first_bad_round").len().sort("first_bad_round", nulls_last=True))
    return d


hdr("Q1 first round where in-game start totals != winner-based totals through previous round")
d305 = first_div(list(set305), "305 set")
dctl = first_div(control, "control d12-25 pc20 (not in 305)")

hdr("Q1b for the control: any divergence at all (count of matches with ANY round 2..13 not direct-ok)")
print("control matches with any non-direct round:", dctl.filter(pl.col("first_bad_round").is_not_null()).height, "of", dctl.height)
print("305 set matches with any non-direct round:", d305.filter(pl.col("first_bad_round").is_not_null()).height, "of", d305.height)

# ----------------------------------------------------------------------------- Q2 round-8 signature
hdr("Q2 round 8 (player_count == 20, d12-25): ADJ side status, 305 set vs control")
r8 = M.filter((pl.col("round") == 8) & (pl.col("dg") == "d12-25") & (pl.col("player_count") == 20)).with_columns(
    pl.col("match_key").is_in(list(set305)).alias("in305"),
    (pl.col("t1") - pl.col("t0")).alias("dt8"), (pl.col("c1") - pl.col("c0")).alias("dc8"))
print(r8.group_by(["in305", "adj_side"]).len().sort(["in305", "adj_side"]))
print("round-8 within-round (dt, dc) patterns for null-side rows, by group:")
print(r8.filter(pl.col("adj_side").is_null()).group_by(["in305", "dt8", "dc8"]).len().sort("len", descending=True).head(12))
print("round-8 null-side rows: 305 set", r8.filter(pl.col("in305") & pl.col("adj_side").is_null()).height,
      "of", r8.filter(pl.col("in305")).height, "| control", r8.filter(~pl.col("in305") & pl.col("adj_side").is_null()).height,
      "of", r8.filter(~pl.col("in305")).height)

# round-8 end state vs swapped winner totals through round 8
w8 = W.filter(pl.col("round") == 9).select(["match_key", "T_prev", "CT_prev"]).rename({"T_prev": "T8", "CT_prev": "CT8"})
s9 = S.filter(pl.col("round") == 9).select(["match_key", pl.col("t0").alias("T9_start"), pl.col("c0").alias("CT9_start")])
j = w8.join(s9, on="match_key", how="inner").with_columns(pl.col("match_key").is_in(list(set305)).alias("in305"))
j = j.with_columns(
    ((pl.col("T9_start") == pl.col("T8")) & (pl.col("CT9_start") == pl.col("CT8"))).alias("direct8"),
    ((pl.col("T9_start") == pl.col("CT8")) & (pl.col("CT9_start") == pl.col("T8"))).alias("swap8"))
print("\nstart-of-round-9 in-game totals vs winner totals through round 8:")
print(j.group_by("in305").agg(pl.len().alias("n"), pl.col("direct8").sum().alias("direct_match"),
                              pl.col("swap8").sum().alias("swapped_match")))

# ----------------------------------------------------------------------------- Q3 overlaps
hdr("Q3 overlaps with the 305 set (d12-25, pc20)")
r13_mm = set(M.filter((pl.col("round") == 13) & (pl.col("dg") == "d12-25") & (pl.col("player_count") == 20)
                      & pl.col("adj_mm"))["match_key"].to_list())
r8_null = set(M.filter((pl.col("round") == 8) & (pl.col("dg") == "d12-25") & (pl.col("player_count") == 20)
                       & pl.col("adj_side").is_null())["match_key"].to_list())
r13_side = set(M.filter((pl.col("round") == 13) & (pl.col("dg") == "d12-25") & (pl.col("player_count") == 20)
                        & pl.col("adj_side").is_not_null())["match_key"].to_list())
print("305 set:", len(set305))
print("305 with a round-13 ADJ disagreement:", len(set305 & r13_mm), "  expected if independent ~", round(len(set305) * len(r13_mm) / len(r13_side)))
print("305 with a round-8 null ADJ side:", len(set305 & r8_null), " of 305; round-8 null share overall:", round(len(r8_null) / max(1, len(set(mk_pc20))), 4))

# ----------------------------------------------------------------------------- Q4 defused+T reason, all rounds
hdr("Q4 bomb events vs reason, all rounds d12-25 pc20 (ADJ not required)")
BS = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "event_type"]) for f in fl("bomb_state")],
               how="vertical_relaxed")
bo = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
      .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs"))
      .with_columns(pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit("defused"))
                    .otherwise(pl.lit("exploded")).alias("bomb")))
X = (M.filter((pl.col("dg") == "d12-25") & (pl.col("player_count") == 20))
     .join(bo.select(["match_key", "round", "bomb"]), on=["match_key", "round"], how="inner"))
print(X.group_by(["bomb", "win_reason_message", "winner_team_code"]).len().sort(["bomb", "len"], descending=[False, True]))
print("bomb-defused rows with winner 2 (T) by round:",
      X.filter((pl.col("bomb") == "defused") & (pl.col("winner_team_code") == 2)).group_by("round").len().sort("round").to_dicts())
