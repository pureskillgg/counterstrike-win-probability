"""Verifier D3-5, probe 8: where do the 41 death-feed (E) vs score-field (C) disagreements sit in 2024?
Read only. Prints to stdout. Reuses the definitions in verify_D3_5_final.py.
"""
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"G:\pureskill\csds-cs2\scripts\swarm2")))
import verify_D3_5_final as V  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
D = V.build("2024")
FMT = V.formats(D)
C = V.score_winners(D)
rows = collections.Counter()
examples = []
for (m, r) in D["A_first"]:
    f = V.truth_flags(D, m, r, C)
    if f["E"] is None or f["C"] is None:
        continue
    grp = "day12_25" if V.day_of(m) >= 12 else "day01_07"
    key = (grp, FMT.get(m, "none"), "E!=C" if f["E"] != f["C"] else "E==C")
    rows[key] += 1
    if f["E"] != f["C"] and len(examples) < 6:
        examples.append((m, r, f["A"], f["C"], f["E"], len(D["DEATH"][(m, r)].get(V.T, set())),
                         len(D["DEATH"][(m, r)].get(V.CT, set())), sorted(D["BOMB"].get((m, r), []))))
for k, v in sorted(rows.items()):
    if k[2] == "E!=C" or k[1] == "5v5":
        print(k, v)
print("examples (match tail, round, A, C, E, deadT, deadCT, bomb):")
for e in examples:
    print("  ", e[0][-12:], e[1:])
