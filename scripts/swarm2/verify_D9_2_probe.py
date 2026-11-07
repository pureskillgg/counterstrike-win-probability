"""verify_D9_2 probe: schemas, key format, build_num distribution (read-only on the 2024 tome)."""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(120)
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"


def files_of(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


def schema(name):
    fd = BASE / "header" if name == "header" else BASE / f"basic_{name}{SUF}"
    return pl.scan_parquet(files_of(fd)).collect_schema()


print("== header schema")
for k, v in schema("header").items():
    print(f"  {k}: {v}")
print("== round_end schema")
for k, v in schema("round_end").items():
    print(f"  {k}: {v}")
print("== bomb_state schema")
for k, v in schema("bomb_state").items():
    print(f"  {k}: {v}")
print("== player_info schema")
for k, v in schema("player_info").items():
    print(f"  {k}: {v}")

H = pl.scan_parquet(files_of(BASE / "header")).select(
    ["key", "match_id", "match_date", "build_num", "game_directory", "rushb_version", "map_name"]
).collect()
print("header rows", H.height)
print("key sample:", H["key"].head(3).to_list())
print("match_date sample:", H["match_date"].head(3).to_list())
print("build_num counts:")
print(H.group_by("build_num").len().sort("build_num"))
H2 = H.with_columns(pl.col("key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"))
print("key-day parse null:", H2.filter(pl.col("day").is_null()).height)
print("day x build_num (matches):")
print(H2.group_by(["day", "build_num"]).len().sort(["day", "build_num"]))
