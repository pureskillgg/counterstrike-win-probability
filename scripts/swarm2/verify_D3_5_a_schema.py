"""D3-5 verifier, step A: schemas, row counts, event types for the 2024 G: tome (read only, projected)."""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_width_chars(220)
TOME = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"


def folder(ch):
    return TOME / ("header" if ch == "header" else f"basic_{ch}{SUF}")


for ch in ["header", "round_end", "round_state", "bomb_state"]:
    files = sorted(glob.glob(str(folder(ch) / "dataframe_*")))
    sch = pl.read_parquet_schema(files[0])
    n = sum(pl.scan_parquet(f).select(pl.len()).collect().item() for f in files)
    print(f"{ch}: files={len(files)} rows={n}")
    print("  schema:", {k: str(v) for k, v in sch.items()})

H = pl.concat([pl.read_parquet(f, columns=["key", "match_date", "t_starters_score_final",
                                           "ct_starters_score_final", "number_of_points", "map_name"])
               for f in sorted(glob.glob(str(folder("header") / "dataframe_*")))])
print("header sample:")
print(H.head(3))
print("match_date dtype:", H.schema["match_date"])
print("header rows:", H.height, "unique keys:", H["key"].n_unique())
print("number_of_points value counts:", H["number_of_points"].value_counts().sort("count", descending=True).head(5).rows())

RS = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "event_type"])
                for f in sorted(glob.glob(str(folder("round_state") / "dataframe_*")))])
print("round_state event types:")
print(RS.group_by("event_type").len().sort("len", descending=True))
