"""Verifier tests 5 for D5-10 (read only; caches in scratchpad only).
X1  Replicate the claim's regime B figures (D5_17 logic, no 8->9 swap): matches, states, differing states, matches.
X2  For the regime-B matches with a difference: first differing label, round count, and boundary classes at 8, 12, 13.
ASSUMPTION A12: regime B round 1 result is taken from freeze-end round 2 (as in D5_10), since round_end label 1 is absent.
Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\verify_D5_10_tests5.py
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40)
SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D5_10")
T, CT = 2, 3
PAIR = pl.read_parquet(SCR / "pairs.parquet")
FE = pl.read_parquet(SCR / "fe.parquet")
RE = pl.read_parquet(SCR / "re.parquet")
MR = pl.read_parquet(SCR / "mr.parquet")
sw8 = PAIR.filter((pl.col("round") == 8) & (pl.col("cls") == "rec_H1")).select("match_key").unique().with_columns(swap8=pl.lit(True))
MR = MR.join(sw8, on="match_key", how="left").with_columns(swap8=pl.col("swap8").fill_null(False))
B = MR.filter((pl.col("regime") == "B") & ~pl.col("swap8")).select("match_key")
print("regime B no-swap8 matches:", B.height)
Q = RE.join(FE, on=["match_key", "round"], how="left").join(B, on="match_key", how="inner")
Q = Q.join(FE.filter(pl.col("round") == 2).select(["match_key", pl.col("t0").alias("t2"), pl.col("c0").alias("c2")]),
           on="match_key", how="left").sort(["match_key", "round"])
onT = (pl.col("round") <= 12) | pl.col("round").is_between(25, 27)
# round_end rows for regime B may lack label 1; FE-only rows (round 1) are not in RE, so build the full round index from FE
R1 = (B.join(FE.filter(pl.col("round") == 2).select(["match_key", pl.col("t0").alias("t2")]), on="match_key", how="inner").with_columns(round=pl.lit(1, dtype=pl.Int64), t0=pl.lit(None, dtype=pl.Int64), c0=pl.lit(None, dtype=pl.Int64)).select(["match_key", "round", "t0", "c0"]))
FULL = (pl.concat([FE.join(B, on="match_key", how="inner").select(["match_key", "round", "t0", "c0"]), R1])
        .join(RE.select(["match_key", "round", "winner_team_code"]), on=["match_key", "round"], how="left")
        .join(FE.filter(pl.col("round") == 2).select(["match_key", pl.col("t0").alias("t2"), pl.col("c0").alias("c2")]),
              on="match_key", how="left").sort(["match_key", "round"]))
FULL = FULL.with_columns(a_fe=pl.when(onT).then(pl.col("t0")).otherwise(pl.col("c0")),
                         a_won=pl.when(pl.col("round") == 1).then(pl.when(onT).then(pl.col("t2")).otherwise(pl.col("c2")))
                         .otherwise(pl.when(onT & (pl.col("winner_team_code") == T)).then(1)
                                    .when((~onT) & (pl.col("winner_team_code") == CT)).then(1).otherwise(0)))
FULL = FULL.with_columns(cum_before=pl.col("a_won").fill_null(0).cum_sum().over("match_key") - pl.col("a_won").fill_null(0))
FULL = FULL.with_columns(diff=pl.col("a_fe") - pl.col("cum_before"))
ck = FULL.filter(pl.col("a_fe").is_not_null() & (pl.col("round") >= 2))
print("regime B round-1 rows added:", FULL.filter(pl.col("round") == 1).height)
print(f"regime B: matches {ck['match_key'].n_unique()}; states {ck.height}; diff!=0 states {ck.filter(pl.col('diff') != 0).height}; "
      f"matches with any diff!=0 {ck.filter(pl.col('diff') != 0)['match_key'].n_unique()}")
dm = ck.filter(pl.col("diff") != 0).group_by("match_key").agg(pl.col("round").min().alias("first_r"), pl.len().alias("n_bad"))
print("first differing label distribution (B):", dm.group_by("first_r").len().sort("first_r").rows())
N = RE.group_by("match_key").agg(pl.col("round").max().alias("N"))
dm = dm.join(N, on="match_key", how="left").join(MR.select(["match_key", "t_starters_score_final", "ct_starters_score_final"]), on="match_key", how="left")
print("N of B diff matches:", sorted(dm["N"].to_list()))
cls = PAIR.filter(pl.col("match_key").is_in(dm["match_key"].to_list()) & pl.col("round").is_in([8, 12, 13])).group_by(["round", "cls"]).len().sort(["round", "cls"])
print("boundary classes at 8, 12, 13 among B diff matches:", cls.rows())
