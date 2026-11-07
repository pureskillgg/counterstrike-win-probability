"""How often does armor wear out mid-round? 150-match 1 Hz sample: alive players only. Armor cannot rise in a round, so the
minimum over a player's alive snapshots is his armor at his last snapshot."""
import os

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import numpy as np
import polars as pl

d = pl.read_parquet(r"G:\pureskill\csds-cs2\data\state_sample\players_1hz.parquet",
                    columns=["match_key", "round", "tick", "seconds_since_live", "player_id", "team_code", "armor", "has_helmet"])
d = d.filter(pl.col("round") <= 24)
g = d.sort("tick").group_by(["match_key", "round", "player_id", "team_code"], maintain_order=True).agg(
    pl.col("armor").first().alias("a0"), pl.col("armor").min().alias("amin"), pl.col("armor").last().alias("alast"),
    pl.len().alias("n"), pl.col("has_helmet").first().alias("helm0"))
arm = g.filter(pl.col("a0") > 0)
print("players with armor at freeze end:", arm.height, "of", g.height)
dep = arm.filter(pl.col("amin") == 0)
print("armor reached 0 while still alive:", dep.height, f"({dep.height / arm.height:.2%} of armored players)")
for thr in (50, 25):
    k = arm.filter(pl.col("amin") < thr).height
    print(f"armor fell below {thr} while alive: {k} ({k / arm.height:.2%})")
lost = (arm["a0"] - arm["amin"]).to_numpy().astype(float)
print("mean armor lost by last alive snapshot:", round(float(lost.mean()), 2), "| median", float(np.median(lost)),
      "| 90th pct", float(np.percentile(lost, 90)))
# among players who survived to 60 s or more
long_ = arm.filter(pl.col("n") >= 60)
print("armored players alive 60+ s:", long_.height, "| armor 0 by then:", long_.filter(pl.col("amin") == 0).height,
      f"({long_.filter(pl.col('amin') == 0).height / max(long_.height, 1):.2%})")
# share of alive armored players with armor < 50 at 30 s and 60 s
for sec in (15, 30, 45, 60, 75):
    x = d.filter((pl.col("seconds_since_live") == sec) & (pl.col("armor") > 0).__or__(pl.col("armor") == 0))
    tot = x.height
    z = x.filter(pl.col("armor") == 0).height
    lo = x.filter(pl.col("armor") < 50).height
    print(f"alive players at {sec:>2} s: {tot:>6} | armor 0: {z / max(tot, 1):.1%} | armor < 50: {lo / max(tot, 1):.1%}")
print("helmet without armor rows at freeze end:", g.filter((pl.col("helm0")) & (pl.col("a0") == 0)).height)
