"""verify D1-6 identity, relative form over both day groups (swarm2 verifier).

Why: round_end has no round-1 rows for d01-07 matches (checked below), so the absolute identity from round 1
cannot be tested there. Relative form: for each match with round_end rounds 2..9 and round_state start of rounds
2 and 10:
  dT  = T10 - T2 (in-game change of the T-labelled counter from start of round 2 to start of round 10)
  dCT = CT10 - CT2
  W_T, W_CT = winner_team_code counts over rounds 2..9
  class = 'direct' if (dT, dCT) == (W_T, W_CT); 'swapped' if == (W_CT, W_T); 'neither' otherwise.
Also reports round-1 presence in round_end by dg, and round-1 presence in round_state by dg.
ASSUMPTIONS: A1 days; A3 2 = T, 3 = CT; A8 duplicate exclusion; A2 not filtered (player_count reported).
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
RS = RS.with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"))
RS = RS.with_columns(pl.when(pl.col("day").is_in(DAYS_HI)).then(pl.lit("d12-25")).otherwise(pl.lit("d01-07")).alias("dg"))

hdr("P1 round-1 presence")
r1_re = RE.filter(pl.col("round") == 1).select(["match_key", "dg"]).unique().group_by("dg").len()
all_m = RE.select(["match_key", "dg"]).unique().group_by("dg").len()
print("round_end matches with round 1 by dg:", r1_re.sort("dg").to_dicts(), " all round_end matches by dg:", all_m.sort("dg").to_dicts())
r1_rs = RS.filter(pl.col("round") == 1).select(["match_key", "dg"]).unique().group_by("dg").len()
print("round_state matches with round 1 by dg:", r1_rs.sort("dg").to_dicts())

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

W = (RE.filter(pl.col("round").is_between(2, 9)).group_by("match_key").agg(
    pl.col("round").n_unique().alias("nr"),
    (pl.col("winner_team_code") == 2).sum().alias("W_T"),
    (pl.col("winner_team_code") == 3).sum().alias("W_CT")))
st2 = S.filter(pl.col("round") == 2).select(["match_key", pl.col("t0").alias("T2"), pl.col("c0").alias("CT2")])
st10 = S.filter(pl.col("round") == 10).select(["match_key", pl.col("t0").alias("T10"), pl.col("c0").alias("CT10")])
meta = RE.select(["match_key", "dg"]).unique()
pc9 = RE.filter(pl.col("round") == 9).select(["match_key", pl.col("player_count").alias("pc9")])
J = (W.filter(pl.col("nr") == 8).join(st2, on="match_key", how="inner").join(st10, on="match_key", how="inner")
     .join(meta, on="match_key", how="left").join(pc9, on="match_key", how="left"))
J = J.with_columns(
    (pl.col("T10") - pl.col("T2")).alias("dT"), (pl.col("CT10") - pl.col("CT2")).alias("dCT"),
    pl.col("match_key").is_in(list(set305)).alias("in305"))
J = J.with_columns(
    pl.when((pl.col("dT") == pl.col("W_T")) & (pl.col("dCT") == pl.col("W_CT"))).then(pl.lit("direct"))
      .when((pl.col("dT") == pl.col("W_CT")) & (pl.col("dCT") == pl.col("W_T"))).then(pl.lit("swapped"))
      .otherwise(pl.lit("neither")).alias("cls"))

hdr("P2 relative identity (rounds 2..9 winners vs in-game change from round-2 start to round-10 start)")
print("matches tested:", J.height)
print(J.group_by(["dg", "in305"]).agg(pl.len().alias("n"), (pl.col("cls") == "direct").sum().alias("direct"),
                                      (pl.col("cls") == "swapped").sum().alias("swapped"),
                                      (pl.col("cls") == "neither").sum().alias("neither")).sort(["dg", "in305"]))

hdr("P3 relative identity, d01-07 by round-9 player_count (is d01-07 clean?)")
print(J.filter(pl.col("dg") == "d01-07").group_by(["pc9", "cls"]).len().sort(["pc9", "cls"]).head(30))

hdr("P4 relative identity, d12-25 control (not in 305) by round-9 player_count")
print(J.filter((pl.col("dg") == "d12-25") & (~pl.col("in305"))).group_by(["pc9", "cls"]).len().sort(["pc9", "cls"]).head(30))
