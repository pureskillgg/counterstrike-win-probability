"""D3-4 verifier probe: event types, header format fields, and swap-claim group sizes (day 12..25)."""
import glob, sys, collections
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(220)
TOME = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
def folder(ch):
    return TOME / "header" if ch == "header" else TOME / f"basic_{ch}{SUF}"
def read(ch, cols=None):
    fs = sorted(glob.glob(str(folder(ch) / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")
RS = read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score", "t_score_raw", "ct_score_raw"])
print("round_state rows:", RS.height, "matches:", RS["match_key"].n_unique())
print("round_state event types:")
print(RS.group_by("event_type").len().sort("len", descending=True))
H = read("header", ["key", "map_name", "match_date", "number_of_points", "match_type", "t_starters_score_final",
                    "ct_starters_score_final", "tick_rate", "match_id"])
print("header rows:", H.height, "unique key:", H["key"].n_unique())
print("sample key:", H["key"][0], "| match_key sample:", RS["match_key"][0])
print("match_type counts:")
print(H.group_by("match_type").len().sort("len", descending=True))
print("number_of_points counts:")
print(H.group_by("number_of_points").len().sort("len", descending=True))
H2 = H.with_columns(pl.col("key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("d")).filter(pl.col("d").cast(pl.Int64) >= 12)
print("day12-25 headers:", H2.height)
print(H2.group_by(["match_type", "number_of_points"]).len().sort("len", descending=True))
print("tick_rate counts:"); print(H2.group_by("tick_rate").len())
