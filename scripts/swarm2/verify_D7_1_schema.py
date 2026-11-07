import glob, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
def files(fd): return sorted(glob.glob(str(fd / "dataframe_*")))
hf = files(BASE / "header")
H = pl.concat([pl.read_parquet(f, columns=["key","match_id","match_date","build_num","demo_version_name","map_name","t_starters_score_final","ct_starters_score_final","number_of_points","match_type","demo_id"]) for f in hf], how="vertical_relaxed")
print("header rows:", H.height)
print("match_date pattern counts:", H.select(pl.col("match_date").str.replace_all(r"\d", "9").alias("p")).group_by("p").len().rows())
print("match_id unique:", H["match_id"].n_unique(), "| key unique:", H["key"].n_unique(), "| demo_id unique:", H["demo_id"].n_unique())
print("build_num:", H["build_num"].value_counts().sort("build_num").rows())
print("match_type:", H["match_type"].value_counts().rows())
print("number_of_points:", H["number_of_points"].value_counts().sort("number_of_points").rows())
rf = files(BASE / f"basic_round_end{SUF}")
print("round_end schema:", pl.read_parquet_schema(rf[0]))
bf = files(BASE / f"basic_bomb_state{SUF}")
print("bomb_state schema:", pl.read_parquet_schema(bf[0]))
