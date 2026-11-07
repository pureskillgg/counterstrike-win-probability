import glob, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
def files(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))
print("header files:", files(BASE / "header"))
hs = pl.read_parquet_schema(files(BASE / "header")[0])
print("header schema:", dict(hs))
H = pl.concat([pl.read_parquet(f) for f in files(BASE / "header")], how="vertical_relaxed")
print("header rows:", H.height)
print(H.head(3))
print("match_date sample:", H["match_date"].head(5).to_list())
for ch in ["round_end", "bomb_state"]:
    fs = files(BASE / f"basic_{ch}{SUF}")
    print(ch, "files", len(fs), "schema", dict(pl.read_parquet_schema(fs[0])))
RE = pl.scan_parquet([str(BASE / f"basic_round_end{SUF}" / Path(f).name) for f in files(BASE / f"basic_round_end{SUF}")])
print(RE.select(pl.len()).collect())
print(RE.select(pl.col("player_count").value_counts()).collect())
print(RE.select(pl.col("match_key").str.extract(r"csds/(\d{4}/\d\d/\d\d)/", 1).value_counts()).collect().sort("match_key"))
