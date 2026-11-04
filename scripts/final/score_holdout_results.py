"""Pin 1b7c86fd, scoring step. Runs only after score_holdout.py has passed the reproduction gate and written p_seed0..2.npy.
Labels are read here for the first time. One read: nothing is recomputed with different settings after this run."""
import json

import numpy as np
import polars as pl

ROOT = "G:/pureskill/csds-cs2"
OCT = f"{ROOT}/data/combined/cache_snapshot_2026_oct23_31.parquet"
HOLD = "C:/Users/<user>/sprint/holdout_oct26_31"
KEY = ["match_key", "round", "tick"]

gate = json.load(open(f"{HOLD}/score_gate.json"))["gate"]
assert all(g["pass"] for g in gate.values()), "gate not passed; no scoring"

keys = pl.read_parquet(f"{HOLD}/keys.parquet")
labels = (pl.scan_parquet(OCT).select(KEY + ["label_ct_win"])
          .filter(pl.col("match_key").is_in(keys["match_key"].unique().to_list()))
          .collect())
J = keys.join(labels, on=KEY, how="left", maintain_order="left")
assert J.height == keys.height and J["label_ct_win"].null_count() == 0, "label join failed"
y = J["label_ct_win"].to_numpy().astype(np.float64)
mk = J["match_key"].to_numpy()
n_rows, n_matches = y.size, len(set(mk.tolist()))

def row_loss(p, y):
    p = np.clip(p, 1e-15, 1.0)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))

L = {}
per_seed = {}
for s in (0, 1, 2):
    p = np.load(f"{HOLD}/p_seed{s}.npy").astype(np.float64)
    assert p.size == n_rows, "prediction row count differs from the keys"
    L[s] = row_loss(p, y)
    per_seed[s] = float(L[s].mean())
seed_mean_ll = float(np.mean(list(per_seed.values())))

# Match-cluster bootstrap of the seed-mean log loss (1,000 resamples, RNG seed 0).
Lbar = (L[0] + L[1] + L[2]) / 3.0
uniq, inv = np.unique(mk, return_inverse=True)
sums = np.bincount(inv, weights=Lbar, minlength=uniq.size)
cnts = np.bincount(inv, minlength=uniq.size).astype(np.float64)
rng = np.random.default_rng(0)
boot = np.empty(1000)
for b in range(1000):
    idx = rng.integers(0, uniq.size, uniq.size)
    boot[b] = sums[idx].sum() / cnts[idx].sum()
ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]

out = {"pin": "1b7c86fd", "matches": n_matches, "seconds": n_rows, "share_with_prediction": 1.0,
       "per_seed_log_loss": per_seed, "seed_mean_log_loss": seed_mean_ll, "seed_mean_ci95_match_cluster": ci,
       "validation_seed_mean_for_reference": 0.450252, "gate": gate}
json.dump(out, open(f"{HOLD}/holdout_score.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ("matches", "seconds", "per_seed_log_loss", "seed_mean_log_loss", "seed_mean_ci95_match_cluster")}, indent=1))
