"""verify D1-6 player_count breakdown (swarm2 verifier).

Round-9 ADJ-vs-winner disagreement by player_count value, 2024 tome (d12-25 and d01-07) and 2026 tome.
Purpose: check that the claimed player_count == 20 filter (2024) does not hide mismatches, and that the 2026
zero is not an artifact of a different player_count convention (2026 has no player_count 20 rounds at round 9).
ASSUMPTIONS: A1 days; A3 2 = T, 3 = CT; A4 ADJ rule; A8 duplicate exclusion.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(30)

BASE24 = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
T26 = Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds")
SUF24 = ".2023-12-28,2024-02-25"
DAYS_HI = [f"{d:02d}" for d in range(12, 26)]


def hdr(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


def run(label, re_files, rs_files, day_split):
    RE = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "winner_team_code", "player_count"])
                    for f in re_files], how="vertical_relaxed")
    dups = RE.group_by(["match_key", "round"]).agg(pl.len().alias("n")).filter(pl.col("n") > 1)
    RE = RE.join(dups.select(["match_key", "round"]), on=["match_key", "round"], how="anti")
    RS = pl.scan_parquet(rs_files).select(["match_key", "round", "tick", "t_score", "ct_score"]).sort(
        ["match_key", "round", "tick"])
    S = (RS.group_by(["match_key", "round"], maintain_order=True).agg(
            pl.col("t_score").first().alias("t0"), pl.col("ct_score").first().alias("c0"),
            pl.col("t_score").last().alias("t1"), pl.col("ct_score").last().alias("c1"))
         .collect(engine="streaming"))
    S = S.with_columns(
        pl.when((pl.col("t1") - pl.col("t0")) == 1).then(pl.lit(2))
          .when((pl.col("c1") - pl.col("c0")) == 1).then(pl.lit(3))
          .otherwise(pl.lit(None, dtype=pl.Int64)).alias("adj_side"))
    A = RE.join(S.select(["match_key", "round", "adj_side"]), on=["match_key", "round"], how="left")
    A = A.with_columns((pl.col("adj_side").is_not_null() & (pl.col("adj_side") != pl.col("winner_team_code"))).alias("mm"))
    if day_split:
        A = A.with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"))
        A = A.with_columns(pl.when(pl.col("day").is_in(DAYS_HI)).then(pl.lit("d12-25")).otherwise(pl.lit("d01-07")).alias("dg"))
    else:
        A = A.with_columns(pl.lit("all").alias("dg"))
    x = A.filter(pl.col("round") == 9)
    print(f"\n-- {label}: round-9 rows by player_count (dg groups)")
    t = (x.group_by(["dg", "player_count"]).agg(
            pl.len().alias("rows"),
            pl.col("adj_side").is_not_null().sum().alias("with_side"),
            pl.col("mm").sum().alias("disagree"))
         .sort(["dg", "rows"], descending=[False, True]))
    print(t.head(40))
    return A


hdr("2024 tome round 9 by player_count")
re24 = fl24 = sorted(glob.glob(str(BASE24 / f"basic_round_end{SUF24}" / "dataframe_*")))
rs24 = sorted(glob.glob(str(BASE24 / f"basic_round_state{SUF24}" / "dataframe_*")))
run("2024", re24, rs24, day_split=True)

hdr("2026 tome round 9 by player_count")
re26 = sorted(glob.glob(str(T26 / "basic_round_end*" / "dataframe_*")))
rs26 = sorted(glob.glob(str(T26 / "basic_round_state*" / "dataframe_*")))
run("2026", re26, rs26, day_split=False)
