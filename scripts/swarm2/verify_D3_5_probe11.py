"""Verifier D3-5, probe 11: 5v5 round-13 round_end flips vs (a) label contiguity (round 1 present or missing),
(b) physical swap at 12 (player_info), (c) bomb outcome agreement. Read only. Prints to stdout.
"""
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"G:\pureskill\csds-cs2\scripts\swarm2")))
import verify_D3_5_final as V  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
D = V.build("2024")
FMT = V.formats(D)
PS = V.phys_swaps(D)
C = V.score_winners(D)
labels = collections.defaultdict(set)
for (m, r) in D["A_first"]:
    labels[m].add(r)
cross = collections.Counter()
for (m, r), a in D["A_first"].items():
    if r != 13 or FMT.get(m) != "5v5" or V.day_of(m) < 12:
        continue
    contiguous = sorted(labels[m]) == list(range(1, len(labels[m]) + 1))
    c13 = C.get((m, 13))
    swap12 = 12 in PS.get(m, [])
    f = V.truth_flags(D, m, 13, C)
    if c13 is None:
        status = "C_ambiguous_or_missing"
    else:
        status = "A!=C" if a != c13 else "A==C"
    cross[("contiguous" if contiguous else "round1_missing", "phys_swap_12" if swap12 else "no_phys_swap_12", status)] += 1
print("5v5 day 12-25, round 13 rows: (label structure, physical swap at 12, A vs unique score winner):")
for k, v in sorted(cross.items()):
    print("  ", k, v)

# Same for the bomb-outcome rows at round 13 (5v5 and all formats): D vs A, by contiguity
bd = collections.Counter()
for (m, r), a in D["A_first"].items():
    if r != 13 or V.day_of(m) < 12:
        continue
    f = V.truth_flags(D, m, 13, C)
    if f["D"] is None:
        continue
    contiguous = sorted(labels[m]) == list(range(1, len(labels[m]) + 1))
    bd[("contiguous" if contiguous else "round1_missing", FMT.get(m), "D==A" if f["D"] == a else "D!=A")] += 1
print("\nround-13 bomb-outcome rows, day 12-25: (label structure, format, D vs A):")
for k, v in sorted(bd.items(), key=lambda kv: str(kv[0])):
    print("  ", k, v)

# day 01-07: label structure, round 13, all formats
d01 = collections.Counter()
for (m, r), a in D["A_first"].items():
    if r != 13 or V.day_of(m) >= 12:
        continue
    c13 = C.get((m, 13))
    if c13 is None:
        continue
    d01[("round1_missing" if not (sorted(labels[m]) == list(range(1, len(labels[m]) + 1))) else "contiguous",
         "A==C" if a == c13 else "A!=C")] += 1
print("\nday 01-07 round 13 (label structure, A vs C):", dict(d01))
