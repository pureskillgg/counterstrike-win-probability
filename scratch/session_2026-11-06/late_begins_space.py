"""Run ON the space (~/venv-dsdk/bin/python late_begins_space.py). Are any bomb_begin_plant events after the planted tick?

The planted tick is the export's round_plant (the same mask the bucket features use). Read-only.
"""
import json
import os

import numpy as np
import polars as pl

FIN = os.path.expanduser("~/adam/builder/final")
EXP = FIN + "/export_clean"
keys = np.array(json.load(open(EXP + "/match_keys.json")), dtype=object)
rm = np.load(EXP + "/round_match.npy").astype(np.int64)
rn = np.load(EXP + "/round_num.npy").astype(np.int64)
plant = np.load(EXP + "/round_plant.npy").astype(np.int64)
R = pl.DataFrame({"match_key": keys[rm].tolist(), "round": rn, "p": plant}).filter(pl.col("p") >= 0)
P = pl.read_parquet(FIN + "/plants_bucket.parquet").with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64))
J = P.join(R, on=["match_key", "round"], how="inner")
print("planted rounds (export round_plant >= 0):", R.height, flush=True)
print("bomb_begin_plant events in those rounds:", J.height, flush=True)
print("events AFTER the planted tick (+1 s):", int((J["tick"] > J["p"] + 64).sum()), flush=True)
print("events BEFORE the planted tick (-1 s):", int((J["tick"] < J["p"] - 64).sum()), flush=True)
