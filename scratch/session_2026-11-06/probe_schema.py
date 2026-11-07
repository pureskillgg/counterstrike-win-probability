import polars as pl
import os
RAW = "/home/sagemaker-user/adam/builder/raw_cache_bucket/"
d = RAW + "csds/2026/06/11/cf7a4c33-7f0a-4e55-a6f0-054d8b4a900c/"
for t in ["player_vector", "player_spawn", "player_death", "player_status", "player_info", "round_state"]:
    df = pl.read_parquet(d + t)
    print("==", t, df.height, "rows")
    print(df.schema)
    print(df.head(3))
