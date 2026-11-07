"""verify D7-6 step b: do round_state freeze_end t_score / ct_score differ from the raw score fields, by build?
Also: freeze_end row counts per match (duplicates), and whether t/c score sums are consistent with round count.
Stdout only. Lazy scans with projection; one table at a time.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_width_chars(220)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"

H = (pl.read_parquet(str(BASE / "header" / "dataframe_00000"), columns=["key", "match_date", "build_num", "map_name"])
       .rename({"key": "match_key"})
       .with_columns(pl.col("match_date").str.slice(0, 19).str.to_datetime("%Y-%m-%dT%H:%M:%S", strict=False).alias("mdt")))

fs = sorted(glob.glob(str(BASE / f"basic_round_state{SUF}" / "dataframe_*")))
RS = (pl.scan_parquet(fs)
        .filter(pl.col("event_type") == "round_freeze_end")
        .select(["match_key", "round", "tick", "t_score", "ct_score", "t_score_raw", "ct_score_raw", "is_warmup"])
        .collect())
print("freeze_end rows:", RS.height, "| distinct (match,round):", RS.select(["match_key", "round"]).unique().height)
print("is_warmup counts:", RS.group_by("is_warmup").len().rows())
print("null t_score:", RS.filter(pl.col("t_score").is_null()).height,
      "| null t_score_raw:", RS.filter(pl.col("t_score_raw").is_null()).height)

X = RS.join(H, on="match_key", how="left")
print("unmatched to header:", X.filter(pl.col("build_num").is_null()).height)
X = X.with_columns(
    (pl.col("t_score") != pl.col("t_score_raw")).alias("t_diff"),
    (pl.col("ct_score") != pl.col("ct_score_raw")).alias("c_diff"),
)
print("t_score != t_score_raw by build:", X.group_by(["build_num", "t_diff"]).len().sort(["build_num", "t_diff"]).rows())
print("ct_score != ct_score_raw by build:", X.group_by(["build_num", "c_diff"]).len().sort(["build_num", "c_diff"]).rows())

# round-level: does t_score equal the team-on-T raw score? Print sample rows of a 9961 match where any diff occurs.
d = X.filter(pl.col("t_diff") | pl.col("c_diff"))
print("rows with any diff:", d.height, "| builds:", d.group_by("build_num").len().rows())
if d.height:
    k = d["match_key"][0]
    print("sample match:", k, "build", d["build_num"][0])
    print(X.filter(pl.col("match_key") == k).sort("round")
          .select(["round", "t_score", "ct_score", "t_score_raw", "ct_score_raw"]).head(30))

# duplicates per (match, round)
dup = RS.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1)
print("(match,round) with >1 freeze_end row:", dup.height)

# compare t_score to t_score_raw by round-number position for 9961 matches
print("diff rows by round (9961 only, top 15):",
      X.filter((pl.col("build_num") == 9961) & (pl.col("t_diff") | pl.col("c_diff")))
       .group_by("round").len().sort("len", descending=True).head(15).rows())
print("diff rows by round (9842 only, top 15):",
      X.filter((pl.col("build_num") == 9842) & (pl.col("t_diff") | pl.col("c_diff")))
       .group_by("round").len().sort("len", descending=True).head(15).rows())
