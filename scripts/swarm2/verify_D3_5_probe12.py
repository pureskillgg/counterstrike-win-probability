"""Verifier D3-5, probe 12: does the claimed day-by-day round-13 mismatch rate track the share of matches whose
round labels start at 1? Also: round-label structure in the 2026 tome. Read only. Prints to stdout.
"""
import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(r"G:\pureskill\csds-cs2\scripts\swarm2")))
import verify_D3_5_final as V  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
D = V.build("2024")
C = V.score_winners(D)
labels = collections.defaultdict(set)
for (m, r) in D["A_first"]:
    labels[m].add(r)
contig = {m: sorted(rs) == list(range(1, len(rs) + 1)) for m, rs in labels.items()}
per_day_all = collections.Counter()
per_day_contig = collections.Counter()
per_day_bomb = collections.Counter()
per_day_bomb_mis = collections.Counter()
for m in D["header"]:
    if V.day_of(m) < 12:
        continue
    per_day_all[V.day_of(m)] += 1
    per_day_contig[V.day_of(m)] += int(contig.get(m, False))
for (m, r), a in D["A_first"].items():
    if r != 13 or V.day_of(m) < 12:
        continue
    f = V.truth_flags(D, m, 13, C)
    if f["D"] is None:
        continue
    per_day_bomb[V.day_of(m)] += 1
    per_day_bomb_mis[V.day_of(m)] += int(f["D"] != a)
days = sorted(per_day_all)
xs, ys = [], []
print("day | matches | share with round 1 present | round-13 bomb rows | mismatch rate (D != A)")
for d in days:
    share = per_day_contig[d] / per_day_all[d]
    rate = per_day_bomb_mis[d] / per_day_bomb[d] if per_day_bomb[d] else float("nan")
    print(f"{d} | {per_day_all[d]:>4} | {share:.3f} | {per_day_bomb[d]:>4} | {rate:.3f}")
    if per_day_bomb[d]:
        xs.append(share)
        ys.append(rate)
n = len(xs)
mx, my = sum(xs) / n, sum(ys) / n
cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
vx = sum((x - mx) ** 2 for x in xs)
vy = sum((y - my) ** 2 for y in ys)
print(f"Pearson r across {n} days (share with round 1 present vs round-13 mismatch rate): {cov / (vx * vy) ** 0.5:.4f}")
print("day 01-07 matches with round 1 present:", sum(1 for m in D["header"] if V.day_of(m) < 12 and contig.get(m)),
      "of", sum(1 for m in D["header"] if V.day_of(m) < 12))

# 2026 label structure
tome26 = V.TOMES["2026"][0]
suf26 = V.TOMES["2026"][1]
RE26 = V.read(tome26, suf26, "round_end", ["match_key", "round"])
lab26 = collections.defaultdict(set)
for m, r in RE26.iter_rows():
    lab26[m].add(int(r))
c26 = collections.Counter(sorted(rs) == list(range(1, len(rs) + 1)) for rs in lab26.values())
print("\n2026 tome: matches with contiguous round labels starting at 1:", c26.get(True, 0), "| not:", c26.get(False, 0))
