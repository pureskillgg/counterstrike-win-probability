import collections
import re
import numpy as np

SEED = 0
src = open("G:/pureskill/csds-cs2/scripts/swarm2/M1_map_inventory.py", encoding="utf-8").read()
start = src.index("def map_holdout(")
end = src.index("def run(tag, cfg):")
ns = {"collections": collections, "np": np, "SEED": SEED}
exec(src[start:end], ns)
map_holdout = ns["map_holdout"]

rng = np.random.default_rng(1)
maps = ["a", "b", "c"]
rows = []
for m_i in range(400):  # matches
    mp = maps[m_i % 3]
    for r_i in range(60):
        strat = int(rng.integers(0, 5))
        p = 0.3 + 0.1 * strat  # true, no map effect
        y = int(rng.random() < p)
        rows.append((f"m{m_i}", y, strat, mp))
print("null (no map effect):", {k: (round(v["ll"], 5) if isinstance(v, dict) else round(v, 5)) for k, v in map_holdout(rows).items() if k != "n"})
# with a true map effect of +0.05 on map 'a'
rows2 = []
for (mk, y, strat, mp) in rows:
    p = 0.3 + 0.1 * strat + (0.05 if mp == "a" else 0.0)
    rows2.append((mk, int(rng.random() < p), strat, mp))
print("true +0.05 map a:", {k: (round(v["ll"], 5) if isinstance(v, dict) else round(v, 5)) for k, v in map_holdout(rows2).items() if k != "n"})
