"""verify D1-6 roster coverage check (swarm2 verifier): player_info and player_death coverage for round 9 rows.

Why: the death-based winner test (verify_D1_6_deaths.py) marks all 305 disagreeing round-9 rows roster_incomplete.
This script reports the per-row roster (T and CT) and victim counts for round 9 (player_count == 20, d12-25),
split by ADJ status, and the same for the agreeing rows, plus round-8 and round-10 for the 305 set.
ASSUMPTIONS: A1 days; A2 player_count == 20; A3 2 = T, 3 = CT; A8 duplicate exclusion.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(30)
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
DAYS_HI = [f"{d:02d}" for d in range(12, 26)]


def fl(ch):
    return sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))


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
M = RE.join(S, on=["match_key", "round"], how="left").with_columns(
    (pl.col("adj_side").is_not_null() & (pl.col("adj_side") != pl.col("winner_team_code"))).alias("mm"))
set305 = set(M.filter((pl.col("round") == 9) & (pl.col("dg") == "d12-25") & (pl.col("player_count") == 20) & pl.col("mm"))
             ["match_key"].to_list())

PI = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "player_id_fixed", "team_code"])
                for f in fl("player_info")], how="vertical_relaxed")
PI = PI.filter(pl.col("player_id_fixed").is_not_null()).unique(["match_key", "round", "player_id_fixed", "team_code"])
roster = PI.group_by(["match_key", "round"]).agg((pl.col("team_code") == 2).sum().alias("rT"),
                                                 (pl.col("team_code") == 3).sum().alias("rCT"),
                                                 pl.len().alias("n_players"))
PD = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "player_id_fixed", "player_team_code"])
                for f in fl("player_death")], how="vertical_relaxed")
PD = PD.filter(pl.col("player_id_fixed").is_not_null()).unique(["match_key", "round", "player_id_fixed"])
vic = PD.group_by(["match_key", "round"]).agg((pl.col("player_team_code") == 2).sum().alias("vT"),
                                              (pl.col("player_team_code") == 3).sum().alias("vCT"))
X = M.join(roster, on=["match_key", "round"], how="left").join(vic, on=["match_key", "round"], how="left")
X = X.with_columns(pl.col("match_key").is_in(list(set305)).alias("in305"))

print("305 set:", len(set305))
for rr in [8, 9, 10]:
    x = X.filter((pl.col("round") == rr) & (pl.col("dg") == "d12-25") & (pl.col("player_count") == 20))
    print(f"\n== round {rr}, d12-25 pc20 ==")
    print(x.group_by("in305").agg(
        pl.len().alias("rows"),
        pl.col("n_players").is_null().sum().alias("no_player_info"),
        pl.col("rT").median().alias("median_rT"), pl.col("rCT").median().alias("median_rCT"),
        pl.col("vT").median().alias("median_vT"), pl.col("vCT").median().alias("median_vCT"),
        pl.col("vT").is_null().sum().alias("no_death_rows")).sort("in305"))
    print(x.group_by(["in305", "rT", "rCT"]).len().sort("len", descending=True).head(6))
