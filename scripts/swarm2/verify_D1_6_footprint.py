"""verify D1-6 roster-footprint cross-tab (swarm2 verifier).

Footprint of a round = (rT, rCT) distinct player_info players per side in that round (2024 tome).
'small' footprint = rT <= 3 or rCT <= 3 (fewer than 4 players on a side); 'full' = both >= 4; 'none' = no player_info.
Cross-tab with ADJ disagreement at round 9 and round 13 (player_count == 20), by day group.
Question: are the 305 round-9 disagreements a subset of the small-footprint matches, and what is the rate inside vs
outside the footprint? Also: how many matches have small footprint at round 9 overall?
ASSUMPTIONS: A1 days; A2 player_count == 20; A3 2 = T, 3 = CT; A8 duplicate exclusion; A13 ASSUMPTION: 'small' cut is
    at 4 players per side, chosen because the control median is 5 per side.
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
PI = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "player_id_fixed", "team_code"])
                for f in fl("player_info")], how="vertical_relaxed")
PI = PI.filter(pl.col("player_id_fixed").is_not_null()).unique(["match_key", "round", "player_id_fixed", "team_code"])
roster = PI.group_by(["match_key", "round"]).agg((pl.col("team_code") == 2).sum().alias("rT"),
                                                 (pl.col("team_code") == 3).sum().alias("rCT"))
X = M.join(roster, on=["match_key", "round"], how="left").with_columns(
    pl.when(pl.col("rT").is_null()).then(pl.lit("none"))
      .when((pl.col("rT") <= 3) | (pl.col("rCT") <= 3)).then(pl.lit("small"))
      .otherwise(pl.lit("full")).alias("fp"))
X = X.filter(pl.col("player_count") == 20)

for rr in [9, 13, 8]:
    x = X.filter((pl.col("round") == rr) & pl.col("adj_side").is_not_null())
    print(f"\n== round {rr}, pc20, ADJ side present ==")
    print(x.group_by(["dg", "fp"]).agg(pl.len().alias("rows"), pl.col("mm").sum().alias("ADJ_disagree"),
                                       (pl.col("mm").sum() / pl.len()).round(4).alias("rate")).sort(["dg", "fp"]))

m9 = X.filter((pl.col("round") == 9) & (pl.col("dg") == "d12-25"))
dis = set(m9.filter(pl.col("mm"))["match_key"].to_list())
small9 = set(m9.filter(pl.col("fp") == "small")["match_key"].to_list())
print("\nround-9 d12-25 pc20: disagreeing matches", len(dis), " small-footprint matches", len(small9),
      " disagreeing AND small:", len(dis & small9))
print("disagreeing rows whose footprint is 'full':", m9.filter(pl.col("mm") & (pl.col("fp") == "full")).height)

m13 = X.filter((pl.col("round") == 13) & (pl.col("dg") == "d12-25") & pl.col("adj_side").is_not_null())
print("round-13 d12-25 pc20: ADJ disagree rows in small footprint:",
      m13.filter(pl.col("mm") & (pl.col("fp") == "small")).height, " in full:",
      m13.filter(pl.col("mm") & (pl.col("fp") == "full")).height, " small rows total:", m13.filter(pl.col("fp") == "small").height)
