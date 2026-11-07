"""D9 probe: key formats, player_count distribution, header value counts (read-only)."""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"

def files_of(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))

def scan(name):
    fd = BASE / name if name == "header" else BASE / f"basic_{name}{SUF}"
    return pl.scan_parquet(files_of(fd))

H = scan("header").collect()
print("header rows", H.height, "unique key", H["key"].n_unique(), "unique match_id", H["match_id"].n_unique())
print(H.select(["key", "match_id", "match_date", "match_type", "platform", "map_name", "tick_rate",
                "demo_version_name", "build_num", "game", "providence", "number_of_points"]).head(5))
for c in ["match_type", "platform", "map_name", "tick_rate", "game", "providence", "number_of_points",
          "build_num", "demo_version_name", "rushb_version", "ppp_version", "match_date", "game_directory"]:
    vc = H.group_by(c).len().sort("len", descending=True)
    print(f"\n-- header.{c}: {vc.height} distinct")
    print(vc.head(12))

RE = scan("round_end").select(["match_key", "round", "winner_team_code", "player_count"]).collect()
print("\nround_end rows", RE.height, "matches", RE["match_key"].n_unique())
print("sample match_key:", RE["match_key"].head(3).to_list())
print("\nround_end.player_count distribution:")
print(RE.group_by("player_count").len().sort("player_count"))
mk = RE.group_by("match_key").agg(pl.col("player_count").n_unique().alias("npc"))
print("matches with >1 distinct player_count across rounds:", mk.filter(pl.col("npc") > 1).height)
print("sample match_key parts:", RE["match_key"].unique().head(3).to_list())
