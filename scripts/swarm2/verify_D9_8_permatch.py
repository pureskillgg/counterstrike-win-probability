"""verify D9-8, part 5: is the build-9961 round-13 round_end disagreement per match or per row?
(2024 tome, read-only). Truth for this check is the next-round score winner T1 (see
verify_D9_8_scores.py T1 and ASSUMPTION A5); a row disagrees when round_end winner != T1 winner.

Also prints the same per-match histogram for round 9 and round 28 (the other build-9961 rounds with
disagreement in verify_D9_8_perround.py) and for round 14 (control).
Output is printed; no files are written under E:.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_width_chars(200)

BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
EXCL = [12, 24, 27, 30, 33, 36, 39, 42, 45, 48]


def files_of(name):
    fd = BASE / "header" if name == "header" else BASE / f"basic_{name}{SUF}"
    return sorted(glob.glob(str(fd / "dataframe_*")))


def lz(name, cols):
    lf = pl.scan_parquet(files_of(name))
    have = lf.collect_schema().names()
    miss = [c for c in cols if c not in have]
    if miss:
        raise SystemExit(f"{name} missing columns {miss}")
    return lf.select(cols)


H = lz("header", ["key", "build_num"]).collect().rename({"key": "match_key"})
RE = lz("round_end", ["match_key", "round", "winner_team_code"]).collect()
RE = RE.unique(subset=["match_key", "round"], keep="first")
FE = lz("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"]).filter(
    pl.col("event_type") == "round_freeze_end").collect()
FE = FE.sort("tick").group_by(["match_key", "round"]).agg(
    pl.col("t_score").first().alias("t_pre"), pl.col("ct_score").first().alias("ct_pre"))
F = FE.sort(["match_key", "round"]).with_columns(
    pl.col("round").shift(-1).over("match_key").alias("next_round"),
    pl.col("t_pre").shift(-1).over("match_key").alias("t_next"),
    pl.col("ct_pre").shift(-1).over("match_key").alias("ct_next"),
).filter(pl.col("next_round") == pl.col("round") + 1).with_columns(
    (pl.col("t_next") - pl.col("t_pre")).alias("dT"),
    (pl.col("ct_next") - pl.col("ct_pre")).alias("dCT"),
).with_columns(
    pl.when((pl.col("dCT") == 1) & (pl.col("dT") == 0)).then(pl.lit(3))
      .when((pl.col("dT") == 1) & (pl.col("dCT") == 0)).then(pl.lit(2))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("t1_winner"),
).filter(~pl.col("round").is_in(EXCL)).select(["match_key", "round", "t1_winner"])

J = (RE.join(H, on="match_key", how="left").join(F, on=["match_key", "round"], how="inner")
       .filter((pl.col("build_num") == 9961) & pl.col("t1_winner").is_not_null()))
J = J.with_columns((pl.col("winner_team_code") != pl.col("t1_winner")).alias("dis"))

for rnd in [13, 9, 28, 14]:
    sub = J.filter(pl.col("round") == rnd)
    per = sub.group_by("match_key").agg(pl.len().alias("n"), pl.col("dis").sum().alias("k"))
    per = per.with_columns((pl.col("k") / pl.col("n")).alias("frac"))
    print(f"\n== round {rnd}, build 9961: matches={per.height}, rows={sub.height}, disagreeing rows={int(sub['dis'].sum())}")
    print("fraction of rows disagreeing, per match (bucket: matches):")
    bucket = per.with_columns(
        pl.when(pl.col("frac") == 0).then(pl.lit("a: 0"))
          .when(pl.col("frac") < 0.5).then(pl.lit("b: (0,0.5)"))
          .when(pl.col("frac") < 1).then(pl.lit("c: [0.5,1)"))
          .otherwise(pl.lit("d: 1")).alias("bucket"))
    print(bucket.group_by("bucket").len().sort("bucket"))
    print("rows per match in this round (n: matches):", per.group_by("n").len().sort("n").head(12).rows())

print("\nDONE")
