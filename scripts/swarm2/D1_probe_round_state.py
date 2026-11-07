import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(60)
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"


def fl(ch):
    return sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))


RS = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "tick", "event_type", "t_score", "ct_score",
                                             "t_score_raw", "ct_score_raw", "phase", "is_warmup", "second"])
                for f in fl("round_state")], how="vertical_relaxed")
print("round_state rows:", RS.height)
print("event types:")
print(RS.group_by("event_type").len().sort("len", descending=True))
print("phases:")
print(RS.group_by("phase").len().sort("len", descending=True).head(15))
ex = RS.filter(pl.col("match_key").str.contains(r"csds/2024/02/12/m_partial")).sort(["round", "tick"])
print(ex.select(["round", "tick", "event_type", "t_score", "ct_score", "t_score_raw", "ct_score_raw", "phase", "second"]).head(60))
