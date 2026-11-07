"""Compact sequence export of the 5v5 training grid, with a full check against the grid before anything is kept.

Stored per second (one row per step):   alive_ct (int8), alive_t (int8)            -- the only per-second facts
Stored per round:                       freeze tick, plant tick (-1 = no plant), label, match id, round number, map id
Derived at load time (exact, checked here against grid_training_v1):
  seconds_since_round_live = k                         (k = second index inside the round)
  bomb_planted             = plant >= 0 and freeze + 64k >= plant
  seconds_since_plant      = (freeze + 64k - plant) / 64 if planted else -1
  ct_alive_minus_t         = alive_ct - alive_t
  w_round                  = 1 / steps in the round

Output: final/export_v2/ (numpy .npy plus json lists), and a written verification report.
"""
import json
import os
import time

import numpy as np
import polars as pl

F = os.path.expanduser("~/adam/builder/final")
OUT = os.path.join(F, "export_v2")
os.makedirs(OUT, exist_ok=True)
t0 = time.time()

G = pl.read_parquet(os.path.join(F, "grid_training_v1.parquet"),
                    columns=["match_key", "round", "second", "alive_ct", "alive_t", "bomb_planted",
                             "seconds_since_round_live", "seconds_since_plant", "label_ct_win", "map_name"])
mk = G.select("match_key").unique().sort("match_key").with_row_index("match_id")
R = (G.select("match_key", "round", "map_name", "label_ct_win").unique(["match_key", "round"])
     .join(mk, on="match_key", how="left").sort(["match_id", "round"]).with_row_index("round_id"))
RT = (pl.read_parquet(os.path.join(F, "training_rounds_v1.parquet"), columns=["match_key", "round", "freeze", "plant_tick", "use"])
      .filter(pl.col("use")).select("match_key", "round", "freeze", "plant_tick").unique(["match_key", "round"]))
R = R.join(RT, on=["match_key", "round"], how="left")
assert R["freeze"].null_count() == 0, "round without a freeze tick"
G = G.join(R.select("match_key", "round", "round_id"), on=["match_key", "round"], how="left").sort(["round_id", "second"])

chk = G.group_by("round_id").agg(pl.col("second").min().alias("lo"), pl.col("second").max().alias("hi"), pl.len().alias("n")).sort("round_id")
assert chk.filter((pl.col("lo") != 0) | (pl.col("hi") != pl.col("n") - 1)).height == 0, "non-contiguous rounds"
n_per = chk["n"].to_numpy().astype(np.int64)
offsets = np.concatenate([[0], np.cumsum(n_per)]).astype(np.int64)
assert offsets[-1] == G.height
R = R.sort("round_id")
maps = sorted(R["map_name"].unique().to_list())
map_id = {m: i for i, m in enumerate(maps)}
freeze = R["freeze"].to_numpy().astype(np.int64)
plant = R["plant_tick"].fill_null(-1).to_numpy().astype(np.int64)
label = R["label_ct_win"].to_numpy().astype(np.uint8)
match_id = R["match_id"].to_numpy().astype(np.int32)
round_num = R["round"].to_numpy().astype(np.int16)
round_map = np.array([map_id[m] for m in R["map_name"].to_list()], dtype=np.uint8)
alive_ct = G["alive_ct"].to_numpy().astype(np.int8)
alive_t = G["alive_t"].to_numpy().astype(np.int8)

# ---- verify the derived columns against the grid, row by row
N = R.height
ridx = np.repeat(np.arange(N, dtype=np.int64), n_per)
k = np.arange(G.height, dtype=np.int64) - offsets[ridx]
ts = freeze[ridx] + 64 * k
pl_ = plant[ridx]
planted = (pl_ >= 0) & (ts >= pl_)
sec_plant = np.where(planted, (ts - pl_) / 64.0, -1.0)
bad = {
    "second": int((G["second"].to_numpy().astype(np.int64) != k).sum()),
    "bomb_planted": int((G["bomb_planted"].to_numpy().astype(np.int64) != planted.astype(np.int64)).sum()),
    "seconds_since_round_live": int((np.abs(G["seconds_since_round_live"].to_numpy() - k) > 1e-6).sum()),
    "seconds_since_plant": int((np.abs(G["seconds_since_plant"].to_numpy() - sec_plant) > 1e-4).sum()),
    "label_ct_win": int((G["label_ct_win"].to_numpy().astype(np.uint8) != label[ridx]).sum()),
}
print("verification mismatches vs grid_training_v1:", bad, flush=True)
assert sum(bad.values()) == 0, "derived columns do not reproduce the grid"

np.save(os.path.join(OUT, "steps_alive.npy"), np.stack([alive_ct, alive_t], axis=1))
np.save(os.path.join(OUT, "offsets.npy"), offsets)
np.save(os.path.join(OUT, "round_freeze.npy"), freeze.astype(np.int32))
np.save(os.path.join(OUT, "round_plant.npy"), plant.astype(np.int32))
np.save(os.path.join(OUT, "round_label.npy"), label)
np.save(os.path.join(OUT, "round_match.npy"), match_id)
np.save(os.path.join(OUT, "round_num.npy"), round_num)
np.save(os.path.join(OUT, "round_map.npy"), round_map)
with open(os.path.join(OUT, "match_keys.json"), "w") as f:
    json.dump(mk["match_key"].to_list(), f)
with open(os.path.join(OUT, "maps.json"), "w") as f:
    json.dump(maps, f)
with open(os.path.join(OUT, "README.txt"), "w") as f:
    f.write(__doc__)
with open(os.path.join(OUT, "verification.json"), "w") as f:
    json.dump({"rows": int(G.height), "rounds": N, "matches": mk.height, "mismatches": bad}, f, indent=1)

total = sum(os.path.getsize(os.path.join(OUT, x)) for x in os.listdir(OUT))
print("steps", G.height, "| rounds", N, "| matches", mk.height, "| maps", maps, flush=True)
print("export v2 MB", round(total / 1e6, 1), "| secs", round(time.time() - t0), flush=True)
for x in sorted(os.listdir(OUT)):
    print(f"  {x}: {os.path.getsize(os.path.join(OUT, x)) / 1e6:.1f} MB", flush=True)
