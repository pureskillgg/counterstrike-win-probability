"""Export the 5v5 training grid as compact per-round sequences (for GRU / LSTM / Transformer) and a flat table (XGBoost).

Input : final/grid_training_v1.parquet (one row per second per round, string match_key)
Output: final/export_v1/
  steps_int8.npy     (S, 3) int8    alive_ct, alive_t, bomb_planted           per second
  steps_sec_live.npy (S,)   float16 seconds_since_round_live                  per second (integers, exact in float16)
  steps_sec_plant.npy(S,)   float32 seconds_since_plant (-1 before the plant)  per second
  offsets.npy        (N+1,) int64   round r covers steps offsets[r]:offsets[r+1]
  round_label.npy    (N,)   uint8   1 = CT wins
  round_match.npy    (N,)   int32   match id (index into match_keys.json)
  round_num.npy      (N,)   int16   round number inside the match
  round_map.npy      (N,)   uint8   map id (index into maps.json)
  match_keys.json, maps.json, README.txt
Derived at load time, not stored: ct_alive_minus_t = alive_ct - alive_t; w_round = 1 / steps per round.
"""
import json
import os
import time

import numpy as np
import polars as pl

F = os.path.expanduser("~/adam/builder/final")
OUT = os.path.join(F, "export_v1")
os.makedirs(OUT, exist_ok=True)
t0 = time.time()

G = pl.read_parquet(os.path.join(F, "grid_training_v1.parquet"),
                    columns=["match_key", "round", "second", "alive_ct", "alive_t", "bomb_planted",
                             "seconds_since_round_live", "seconds_since_plant", "label_ct_win", "map_name"])
print("grid rows", G.height, "rounds?", flush=True)

mk = G.select("match_key").unique().sort("match_key").with_row_index("match_id")
R = (G.select("match_key", "round", "map_name", "label_ct_win").unique(["match_key", "round"])
     .join(mk, on="match_key", how="left").sort(["match_id", "round"]).with_row_index("round_id"))
print("rounds", R.height, "matches", mk.height, flush=True)
G = G.join(R.select("match_key", "round", "round_id"), on=["match_key", "round"], how="left").sort(["round_id", "second"])

# every round must be a contiguous run of seconds starting at 0
chk = G.group_by("round_id").agg(pl.col("second").min().alias("lo"), pl.col("second").max().alias("hi"), pl.len().alias("n"))
bad = chk.filter((pl.col("lo") != 0) | (pl.col("hi") != pl.col("n") - 1)).height
assert bad == 0, f"{bad} rounds are not contiguous seconds from 0"
n_per = chk.sort("round_id")["n"].to_numpy().astype(np.int64)
offsets = np.concatenate([[0], np.cumsum(n_per)]).astype(np.int64)
assert offsets[-1] == G.height

maps = sorted(R["map_name"].unique().to_list())
map_id = {m: i for i, m in enumerate(maps)}
R = R.sort("round_id")
np.save(os.path.join(OUT, "offsets.npy"), offsets)
np.save(os.path.join(OUT, "round_label.npy"), R["label_ct_win"].to_numpy().astype(np.uint8))
np.save(os.path.join(OUT, "round_match.npy"), R["match_id"].to_numpy().astype(np.int32))
np.save(os.path.join(OUT, "round_num.npy"), R["round"].to_numpy().astype(np.int16))
np.save(os.path.join(OUT, "round_map.npy"), np.array([map_id[m] for m in R["map_name"].to_list()], dtype=np.uint8))

np.save(os.path.join(OUT, "steps_int8.npy"),
        np.stack([G["alive_ct"].to_numpy(), G["alive_t"].to_numpy(), G["bomb_planted"].to_numpy()], axis=1).astype(np.int8))
np.save(os.path.join(OUT, "steps_sec_live.npy"), G["seconds_since_round_live"].to_numpy().astype(np.float16))
np.save(os.path.join(OUT, "steps_sec_plant.npy"), G["seconds_since_plant"].to_numpy().astype(np.float32))

with open(os.path.join(OUT, "match_keys.json"), "w") as f:
    json.dump(mk["match_key"].to_list(), f)
with open(os.path.join(OUT, "maps.json"), "w") as f:
    json.dump(maps, f)
with open(os.path.join(OUT, "README.txt"), "w") as f:
    f.write(__doc__)

total = sum(os.path.getsize(os.path.join(OUT, x)) for x in os.listdir(OUT))
print("steps", G.height, "| rounds", R.height, "| matches", mk.height, "| maps", maps, flush=True)
print("max steps per round", int(n_per.max()), "| mean", round(float(n_per.mean()), 1), flush=True)
print("export MB", round(total / 1e6, 1), "| secs", round(time.time() - t0), flush=True)
for x in sorted(os.listdir(OUT)):
    print(f"  {x}: {os.path.getsize(os.path.join(OUT, x)) / 1e6:.1f} MB", flush=True)
