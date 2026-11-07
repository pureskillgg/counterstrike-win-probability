"""verify D1-6 probe2 (swarm2 verifier): duplicates, event types, raw vs adjusted score, match_key shape."""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(80)
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"


def fl(ch):
    return sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))


RE = pl.read_parquet(fl("round_end")[0])
print("round_end rows", RE.height, "distinct match_key", RE["match_key"].n_unique())
print("match_key sample:", RE["match_key"].head(3).to_list())
dup = RE.group_by(["match_key", "round"]).agg(pl.len().alias("n")).filter(pl.col("n") > 1)
print("round_end duplicate (match_key, round) groups:", dup.height)
if dup.height:
    print(dup)
    print("dup by round:", dup["round"].to_list())
print("player_count dist:", RE.group_by("player_count").len().sort("len", descending=True).head(6).to_dicts())
print("round_end round range:", RE["round"].min(), RE["round"].max())

RS = pl.read_parquet(fl("round_state")[0])
print("\nround_state rows", RS.height)
print("event_type dist:", RS.group_by("event_type").len().sort("len", descending=True).to_dicts())
print("phase dist:", RS.group_by("phase").len().sort("len", descending=True).head(8).to_dicts())
print("is_warmup dist:", RS.group_by("is_warmup").len().to_dicts())
diff_raw = RS.filter((pl.col("t_score") != pl.col("t_score_raw")) | (pl.col("ct_score") != pl.col("ct_score_raw")))
print("rows where adjusted != raw score:", diff_raw.height)
print(diff_raw.head(5))
print("round_state round range:", RS["round"].min(), RS["round"].max())
