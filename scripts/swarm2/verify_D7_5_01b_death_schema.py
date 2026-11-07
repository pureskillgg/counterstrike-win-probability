"""verify D7-5 probe: basic_player_death schema and event counts (prints only)."""
import glob, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
fs = sorted(glob.glob(str(BASE / f"basic_player_death{SUF}" / "dataframe_*")))
print("files", len(fs), "schema", dict(pl.read_parquet_schema(fs[0])))
lf = pl.concat([pl.scan_parquet(f) for f in fs])
print("rows", lf.select(pl.len()).collect().item())
print(lf.select(["round", "tick", "match_key"]).head(3).collect())
print(lf.group_by("event_type").len().collect() if "event_type" in pl.read_parquet_schema(fs[0]) else "no event_type")
