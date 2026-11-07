import polars as pl
import os
RAW = "/home/sagemaker-user/adam/builder/raw_cache_bucket/"
d = RAW + "csds/2026/06/11/m_partial/"
for t in ["player_vector", "player_spawn", "player_death", "player_status", "player_info", "round_state"]:
    df = pl.read_parquet(d + t)
    print("==", t, df.height, "rows")
    print(df.schema)
    print(df.head(3))
