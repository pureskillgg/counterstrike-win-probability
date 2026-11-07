"""verify D9-9, step 4: which round_state event types carry score fields, by build (read-only).

Shows, for a build-9961 match and a build-9842 match, every round_state row with a score value
in rounds 11-14 (all event types), and the event-type coverage per build.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(120)
pl.Config.set_tbl_cols(12)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"


def files_of(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


H = pl.scan_parquet(files_of(BASE / "header")).select(["key", "build_num"]).collect().rename({"key": "match_key"})
RS = pl.scan_parquet(files_of(BASE / f"basic_round_state{SUF}")).select(
    ["match_key", "round", "tick", "event_type", "t_score", "ct_score"]).collect()
RS = RS.join(H, on="match_key", how="left")

cov = RS.group_by(["build_num", "event_type"]).len().sort(["event_type", "build_num"])
print("== event-type coverage by build (rows)")
print(cov)

for b in [9961, 9842]:
    k = H.filter(pl.col("build_num") == b).select("match_key").head(1)["match_key"].item()
    print(f"\n== build {b} example {k}: rows with scores, rounds 11-14, all events")
    x = RS.filter((pl.col("match_key") == k) & pl.col("round").is_between(11, 14)).sort("tick")
    print(x.select(["round", "tick", "event_type", "t_score", "ct_score"]))
