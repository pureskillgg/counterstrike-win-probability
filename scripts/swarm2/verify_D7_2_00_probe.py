"""verify D7-2 probe: raw round_end win reasons, header build/map layout, first build-9961 match. ASSUMPTION: none beyond the header schema."""
import sys, glob
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(240)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
def files(fd): return sorted(glob.glob(str(fd / "dataframe_*")))
re_files = files(BASE / f"basic_round_end{SUF}")
print("round_end files:", len(re_files))
RE = pl.concat([pl.scan_parquet(f).select(["match_key", "round", "tick", "winner_team_code", "win_reason_code", "win_reason_message"]) for f in re_files]).collect()
print("round_end rows:", RE.height, "| distinct matches:", RE["match_key"].n_unique())
print(RE.group_by(["win_reason_code", "win_reason_message", "winner_team_code"]).len().sort(["win_reason_code", "winner_team_code"]))
H = pl.read_parquet(str(BASE / "header" / "dataframe_00000"), columns=["key", "match_date", "build_num", "map_name", "server_name", "tick_rate"])
print("header rows:", H.height, "| distinct key:", H["key"].n_unique())
print("map_name counts:", H.group_by("map_name").len().sort("len", descending=True).rows())
print("build_num x map:", H.group_by(["build_num", "map_name"]).len().sort(["build_num", "map_name"]).rows())
print("tick_rate:", H.group_by("tick_rate").len().rows())
print("match_date formats sample:", H["match_date"].head(3).to_list(), "| non-Z count:", H.filter(~pl.col("match_date").str.ends_with("Z")).height)
