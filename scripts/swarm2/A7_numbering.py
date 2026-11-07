"""A7: round numbering. For each freeze row, t + c should equal round - 1 (scores before the round).
Checks 2026 and 2024, and the 2024 matches with round 1 missing versus present."""
import collections, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
import importlib.util
ROOT = Path(r"G:\pureskill\csds-cs2")
spec = importlib.util.spec_from_file_location("A7", ROOT / "scripts" / "swarm2" / "A7_score_coverage.py")
A7 = importlib.util.module_from_spec(spec); spec.loader.exec_module(A7)
for label in ("2026", "2024"):
    H, W, F, info = A7.load_tome(label, *A7.TOMES[label])
    tot = collections.Counter()
    for m, Fm in F.items():
        has1 = 1 in Fm or 1 in W.get(m, {})
        for r, (t, c) in Fm.items():
            ok = (t + c == r - 1)
            tot[("round1_present" if has1 else "round1_missing", "sum==r-1" if ok else "sum!=r-1")] += 1
    print(label, dict(sorted(tot.items())))
    # the minimum freeze round in matches missing round 1: is t+c == r-1 there too?
    mins = collections.Counter()
    for m, Fm in F.items():
        if 1 not in Fm and Fm:
            r = min(Fm); t, c = Fm[r]
            mins[(r, t + c == r - 1)] += 1
    print(label, "min freeze round in matches without round 1 (round, sum==r-1):", dict(mins.most_common(6)))
