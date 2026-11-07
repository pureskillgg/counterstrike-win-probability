"""verify D9-10 (d) probe: round_state event types and score fields for build 9961 (read-only, lazy)."""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(12)
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
fs = sorted(glob.glob(str(BASE / f"basic_round_state{SUF}" / "dataframe_*")))
RS = pl.scan_parquet(fs).select(["match_key", "round", "tick", "event_type", "t_score", "ct_score",
                                  "t_score_raw", "ct_score_raw", "phase", "is_warmup"]).collect()
print("round_state rows:", RS.height)
print(RS.group_by("event_type").len().sort("len", descending=True).rows())
print(RS.group_by("phase").len().sort("len", descending=True).head(10).rows())
print("is_warmup counts:", RS.group_by("is_warmup").len().rows())
ex = RS.filter(pl.col("event_type") == "round_freeze_end").head(3)
print(ex)
