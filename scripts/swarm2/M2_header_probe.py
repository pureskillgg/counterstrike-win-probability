"""M2 probe: header schema and map distribution for both tomes (read-only, header only)."""
import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path("G:/pureskill/csds-cs2")
TOMES = {
    "2026": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds" / "header",
    "2024": ROOT / "tomes" / "2024-02-g" / "tome" / "csds" / "header",
}

for era, folder in TOMES.items():
    files = sorted(glob.glob(str(folder / "dataframe_*")))
    print(f"== {era}: {len(files)} header pages in {folder}")
    lf = pl.concat([pl.scan_parquet(f) for f in files], how="vertical_relaxed")
    names = lf.collect_schema().names()
    cols = ["key", "match_id", "match_date", "map_name", "tick_rate", "match_type",
            "t_starters_score_final", "ct_starters_score_final"]
    if "max_rounds" in names:
        cols.append("max_rounds")
    df = lf.select(cols).collect()
    print("rows:", df.height, "unique key:", df["key"].n_unique(), "unique match_id:", df["match_id"].n_unique())
    print("map_name nulls:", df["map_name"].null_count())
    print("match_type:", df.group_by("match_type").agg(pl.len()).rows())
    print("tick_rate:", df.group_by("tick_rate").agg(pl.len()).rows())
    vc = df.group_by("map_name").agg(pl.len().alias("n")).sort("n", descending=True).rows()
    for m, n in vc:
        print(f"  {m!s:<16} {n}")
    print("match_date min/max:", df["match_date"].min(), df["match_date"].max())
    if "max_rounds" in cols:
        print("max_rounds dist:", df.group_by("max_rounds").agg(pl.len()).sort("max_rounds").rows())
    print()
