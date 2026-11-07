"""verify D1-6 identity, clean rebuild over both day groups (swarm2 verifier).

For every match with round_end rounds 1..10 and round_state rows for rounds 1..10:
  W_T, W_CT   = counts of winner_team_code 2 and 3 over rounds 1..9 (round_end, dedup A8)
  T10, CT10   = in-game adjusted score at the first round_state row of round 10 (t_score, ct_score)
  class       = 'direct' if (T10, CT10) == (W_T, W_CT); 'swapped' if == (W_CT, W_T); 'neither' otherwise.
Reported by dg (d01-07, d12-25) x in-set (305 set), and by player_count (20 / other) for d12-25.
Also: the same at round 9 start (totals through round 8) for context.
ASSUMPTIONS: A1 days; A3 2 = T, 3 = CT; A8 duplicate exclusion; A2 player_count reported separately, not filtered.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(30)

BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
DAYS_HI = [f"{d:02d}" for d in range(12, 26)]


def fl(ch):
    return sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))


def hdr(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


RE = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "winner_team_code", "player_count"])
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
M = M.with_columns((pl.col("adj_side").is_not_null() & (pl.col("adj_side") != pl.col("winner_team_code"))).alias("mm"))

set305 = set(M.filter((pl.col("round") == 9) & (pl.col("dg") == "d12-25") & (pl.col("player_count") == 20) & pl.col("mm"))
             ["match_key"].to_list())
print("305 set:", len(set305))

# per-match identity at round 10 start (needs winners 1..9 and in-game round 10 start)
W19 = (RE.filter(pl.col("round") <= 9).group_by("match_key").agg(
    pl.col("round").n_unique().alias("nr"),
    (pl.col("winner_team_code") == 2).sum().alias("W_T"),
    (pl.col("winner_team_code") == 3).sum().alias("W_CT")))
meta = RE.select(["match_key", "dg"]).unique()
pc20 = RE.filter(pl.col("round") == 9).select(["match_key", pl.col("player_count").alias("pc9")])
R10 = S.filter(pl.col("round") == 10).select(["match_key", pl.col("t0").alias("T10"), pl.col("c0").alias("CT10")])
J = (W19.filter(pl.col("nr") == 9).join(R10, on="match_key", how="inner")
     .join(meta, on="match_key", how="left").join(pc20, on="match_key", how="left"))
J = J.with_columns(
    pl.col("match_key").is_in(list(set305)).alias("in305"),
    pl.when((pl.col("T10") == pl.col("W_T")) & (pl.col("CT10") == pl.col("W_CT"))).then(pl.lit("direct"))
      .when((pl.col("T10") == pl.col("W_CT")) & (pl.col("CT10") == pl.col("W_T"))).then(pl.lit("swapped"))
      .otherwise(pl.lit("neither")).alias("cls"))
print("matches with rounds 1..9 and round-10 start:", J.height)

hdr("I1 identity at round-10 start by dg (all player counts)")
print(J.group_by("dg").agg(pl.len().alias("n"), (pl.col("cls") == "direct").sum().alias("direct"),
                           (pl.col("cls") == "swapped").sum().alias("swapped"),
                           (pl.col("cls") == "neither").sum().alias("neither")).sort("dg"))

hdr("I2 identity at round-10 start by dg x in305 (305 set is d12-25 pc20 only)")
print(J.group_by(["dg", "in305"]).agg(pl.len().alias("n"), (pl.col("cls") == "direct").sum().alias("direct"),
                                      (pl.col("cls") == "swapped").sum().alias("swapped"),
                                      (pl.col("cls") == "neither").sum().alias("neither")).sort(["dg", "in305"]))

hdr("I3 identity at round-10 start, d12-25, by round-9 player_count (20 vs other)")
J3 = J.filter(pl.col("dg") == "d12-25").with_columns(
    pl.when(pl.col("pc9") == 20).then(pl.lit("pc20")).otherwise(pl.lit("pc_other")).alias("pcg"))
print(J3.group_by(["pcg", "cls"]).len().sort(["pcg", "cls"]))

hdr("I4 identity at round-10 start, d01-07 by round-9 player_count")
J4 = J.filter(pl.col("dg") == "d01-07")
print(J4.group_by(["pc9", "cls"]).len().sort(["pc9", "cls"]).head(30))

hdr("I5 d01-07: first 12 matches with class neither (for inspection)")
print(J4.filter(pl.col("cls") == "neither").select(["match_key", "T10", "CT10", "W_T", "W_CT", "pc9"]).head(12))
