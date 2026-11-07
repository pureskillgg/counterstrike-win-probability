"""verify D6-6, step 10: 2026 control header build_num / server / map distribution (is 2026 a build-mixed control?)."""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(220)
B26 = Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds")
H = pl.concat([pl.read_parquet(f, columns=["key", "match_date", "build_num", "server_name", "map_name", "max_rounds"])
               for f in sorted(glob.glob(str(B26 / "header" / "dataframe_*")))], how="vertical_relaxed")
print("2026 headers:", H.height, "| build_num distribution:", H.group_by("build_num").len().sort("build_num").rows())
print("2026 max_rounds x build_num:", H.group_by(["max_rounds", "build_num"]).len().sort(["max_rounds", "build_num"]).rows())
print("2026 match_date range:", H["match_date"].min(), "..", H["match_date"].max())
print("2026 server groups:", H.with_columns(pl.when(pl.col("server_name").str.contains("FACEIT")).then(pl.lit("FACEIT"))
                                           .otherwise(pl.lit("Valve")).alias("g")).group_by("g").len().rows())
