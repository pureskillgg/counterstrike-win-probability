"""Verifier probe 3: schemas for player_death, header build fields, tick_rate. Read-only."""
import glob
import sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
fs = sorted(glob.glob(str(BASE / f"basic_player_death{SUF}" / "dataframe_*")))
print("player_death files", len(fs), "cols", list(pl.read_parquet_schema(fs[0]).keys()))
pd_ = pl.read_parquet(fs[0]).head(3)
print(pd_.select([c for c in pd_.columns if "team" in c or c in ("round", "tick", "match_key", "victim_id", "player_id_fixed")]))
hf = sorted(glob.glob(str(BASE / "header" / "dataframe_*")))
H = pl.read_parquet(hf[0], columns=["key", "build_num", "demo_version_name", "tick_rate", "server_name", "network_protocol", "platform", "match_date"]).rename({"key": "match_key"})
print(H.group_by(["build_num", "demo_version_name", "tick_rate", "platform"]).len().sort("build_num"))
print(H.group_by(H["match_date"].str.slice(0, 10)).agg(pl.col("build_num").n_unique().alias("nb"), pl.col("build_num").min().alias("bmin"), pl.col("build_num").max().alias("bmax")).sort("match_date").filter(pl.col("match_date") >= "2024-02-03").head(12))
