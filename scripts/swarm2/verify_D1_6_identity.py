"""verify D1-6 identity + bomb-row check (swarm2 verifier).

Test 1 (cumulative identity): in-game score at the first round_state row of round 10 should equal the number of
  rounds 1..9 won by each side. Compare the identity under two winner sources:
    W = round_end.winner_team_code (2 = T, 3 = CT), counted over rounds 1..9;
    S = score side of rounds 1..9 (ADJ rule, as cited), counted over rounds 1..9.
  Restricted to matches where all of rounds 1..10 exist in round_end and round_state. Reported for all matches
  (d01-07 and d12-25, any player_count) and for the 305 round-9 disagreement matches.
Test 2 (bomb rows): the 10 round-9 d12-25 rows with a bomb outcome that disagrees with winner_team_code.
  For each: winner, win reason, score side, bomb event tick vs the round's round_end tick and window.

ASSUMPTIONS: A1 days from match_key; A2 player_count==20 used only to define the 305 set (as cited);
A8 duplicate (match, round) round_end groups excluded (1 group, round 1 of m_56c0f3b3...).
A9 ASSUMPTION: team codes are side-based across rounds 1..10 (no halftime swap before round 13).
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
                                            "win_reason_message", "tick"])
                for f in fl("round_end")], how="vertical_relaxed")
dups = RE.group_by(["match_key", "round"]).agg(pl.len().alias("n")).filter(pl.col("n") > 1)
RE = RE.join(dups.select(["match_key", "round"]), on=["match_key", "round"], how="anti")
RE = RE.with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"))
RE = RE.with_columns(pl.when(pl.col("day").is_in(DAYS_HI)).then(pl.lit("d12-25")).otherwise(pl.lit("d01-07")).alias("dg"))

RS = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
                for f in fl("round_state")], how="vertical_relaxed").sort(["match_key", "round", "tick"])
S = RS.group_by(["match_key", "round"], maintain_order=True).agg(
    pl.col("t_score").first().alias("t0"), pl.col("ct_score").first().alias("c0"),
    pl.col("t_score").last().alias("t1"), pl.col("ct_score").last().alias("c1"),
    pl.col("tick").min().alias("tick_first"), pl.col("tick").max().alias("tick_last"))
S = S.with_columns(
    pl.when((pl.col("t1") - pl.col("t0")) == 1).then(pl.lit(2))
      .when((pl.col("c1") - pl.col("c0")) == 1).then(pl.lit(3))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("adj_side"))

M = RE.join(S, on=["match_key", "round"], how="left")
M = M.with_columns(
    pl.when(pl.col("adj_side").is_null()).then(pl.lit("no_side"))
      .when(pl.col("adj_side") == pl.col("winner_team_code")).then(pl.lit("agree"))
      .otherwise(pl.lit("disagree")).alias("status"))

# matches with round 9 disagreement (player_count == 20, d12-25) = the claim's 305 set
set305 = (M.filter((pl.col("round") == 9) & (pl.col("dg") == "d12-25") & (pl.col("player_count") == 20)
                   & (pl.col("status") == "disagree"))["match_key"].unique().to_list())
print("305 set size:", len(set305))

# ----------------------------------------------------------------------------- Test 1 identity
# need round 1..10 present for a match; in-game score at round-10 first row
rounds_needed = list(range(1, 11))
cnt = M.filter(pl.col("round").is_in(list(range(1, 10)))).group_by("match_key").agg(
    pl.col("round").n_unique().alias("n_rounds_1_9"),
    (pl.col("winner_team_code") == 2).sum().alias("W_T"),
    (pl.col("winner_team_code") == 3).sum().alias("W_CT"),
    (pl.col("adj_side") == 2).sum().alias("S_T"),
    (pl.col("adj_side") == 3).sum().alias("S_CT"),
    pl.col("adj_side").is_null().sum().alias("n_noside_1_9"))
r10 = S.filter(pl.col("round") == 10).select(["match_key", pl.col("t0").alias("T10"), pl.col("c0").alias("CT10")])
J = cnt.join(r10, on="match_key", how="inner").filter(pl.col("n_rounds_1_9") == 9)
J = J.join(M.select(["match_key", "dg"]).unique(), on="match_key", how="left")
J = J.with_columns(pl.col("match_key").is_in(set305).alias("in305"))
J = J.with_columns(
    ((pl.col("T10") == pl.col("W_T")) & (pl.col("CT10") == pl.col("W_CT"))).alias("id_winner"),
    ((pl.col("T10") == pl.col("S_T")) & (pl.col("CT10") == pl.col("S_CT"))).alias("id_score"))

hdr("T1a identity at round 10 start, all matches with rounds 1..10 present (winner-based vs score-based counts)")
print("matches tested:", J.height)
print(J.group_by("dg").agg(pl.len().alias("n"), pl.col("id_winner").sum().alias("winner_identity_holds"),
                           pl.col("id_score").sum().alias("score_identity_holds")).sort("dg"))

hdr("T1b identity at round 10 start, the 305 round-9 disagreement matches")
J305 = J.filter(pl.col("in305"))
print("matches tested:", J305.height)
print("winner-based identity holds:", int(J305["id_winner"].sum()), " score-based identity holds:", int(J305["id_score"].sum()))
print("both hold:", int((J305["id_winner"] & J305["id_score"]).sum()), " neither:",
      int((~J305["id_winner"] & ~J305["id_score"]).sum()))
print("example mismatch rows (T10, CT10 vs W_T, W_CT, S_T, S_CT):")
print(J305.select(["match_key", "T10", "CT10", "W_T", "W_CT", "S_T", "S_CT"]).head(8))

hdr("T1c same identity, matches NOT in the 305 set, d12-25 (control)")
Jc = J.filter((~pl.col("in305")) & (pl.col("dg") == "d12-25"))
print("matches tested:", Jc.height, " winner identity holds:", int(Jc["id_winner"].sum()),
      " score identity holds:", int(Jc["id_score"].sum()))

# ----------------------------------------------------------------------------- Test 2 bomb rows
BS = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "tick", "event_type"]) for f in fl("bomb_state")],
               how="vertical_relaxed")
bo = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
      .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs"),
                                             pl.col("tick").min().alias("bomb_tick"))
      .with_columns(pl.when(pl.col("evs").list.contains("bomb_defused")).then(3).otherwise(2).alias("bomb_side")))
bo_r9 = bo.filter(pl.col("round") == 9).select(["match_key", "evs", "bomb_side", "bomb_tick"])
bomb_rows = (M.filter((pl.col("round") == 9) & (pl.col("dg") == "d12-25") & (pl.col("player_count") == 20)
                      & (pl.col("status") == "disagree"))
             .join(bo_r9, on="match_key", how="inner"))
hdr("T2 round-9 disagreement rows that also have a bomb outcome (claimed 10)")
print("rows:", bomb_rows.height)
print(bomb_rows.select(["match_key", "winner_team_code", "adj_side", "bomb_side", "evs", "win_reason_message",
                        "tick", "tick_first", "tick_last", "bomb_tick"]))

hdr("T2b for the same rows: is the bomb tick inside the round_state window of round 9?")
print(bomb_rows.with_columns(
    ((pl.col("bomb_tick") >= pl.col("tick_first")) & (pl.col("bomb_tick") <= pl.col("tick_last"))).alias("in_window"))
      .select(["match_key", "in_window", "bomb_tick", "tick_first", "tick_last", "tick"]))

hdr("T2c bomb-outcome rows among AGREE round-9 d12-25 rows: reason vs bomb (base rate for the reason/bomb pairing)")
agr = (M.filter((pl.col("round") == 9) & (pl.col("dg") == "d12-25") & (pl.col("player_count") == 20)
                & (pl.col("status") == "agree"))
       .join(bo_r9, on="match_key", how="inner"))
print(agr.group_by(["win_reason_message", "bomb_side"]).len().sort("len", descending=True))
