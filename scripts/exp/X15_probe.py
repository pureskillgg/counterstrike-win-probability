import os
os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
import sys
import polars as pl
import numpy as np
import xgboost as xgb
import sklearn

print("python", sys.version)
print("polars", pl.__version__, "numpy", np.__version__, "xgboost", xgb.__version__, "sklearn", sklearn.__version__)

base = "G:/pureskill/csds-cs2/data/state_sample/"
p = pl.read_parquet(base + "players_1hz.parquet")
r = pl.read_parquet(base + "rounds.parquet")
print("players shape", p.shape)
print(p.schema)
print("rounds shape", r.shape)
print(r.schema)
print(p.head(8))
print(r.head(5))
print("n matches players", p["match_key"].n_unique(), "rounds", r["match_key"].n_unique())
print("team_code counts", p.group_by("team_code").len().sort("team_code"))
print("split counts", r.group_by("split").len())
print("seconds_since_live describe", p.select(pl.col("seconds_since_live").min().alias("min"), pl.col("seconds_since_live").max().alias("max")))
print("inv_primary value counts top", p.group_by("inv_primary").len().sort("len", descending=True).head(40))
