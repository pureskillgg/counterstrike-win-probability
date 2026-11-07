"""A7: per-match boundary pattern signatures (score-only) for 2026 and 2024. No winner codes used.

Symbol per boundary r -> r+1 (both freeze rows present):
  n = noswap-only, s = swap-only, a = ambiguous (t == c type tie), x = neither.
Pattern = the boundary symbols concatenated for rounds 1..max. Also the list of swap positions
(boundaries classified as swap-only) and the list of freeze rounds that are missing.
"""
import collections
import importlib.util
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
spec = importlib.util.spec_from_file_location("A7", ROOT / "scripts" / "swarm2" / "A7_score_coverage.py")
A7 = importlib.util.module_from_spec(spec); spec.loader.exec_module(A7)

SYM = {"noswap": "n", "swap": "s", "ambig": "a", "neither": "x"}

def sig_for(Fm):
    rs = sorted(Fm)
    if not rs:
        return "", []
    mx = rs[-1]
    out, swaps = [], []
    for r in range(1, mx):
        if r in Fm and r + 1 in Fm:
            t, c = Fm[r]; t2, c2 = Fm[r + 1]
            cls = A7.boundary_class(t, c, t2, c2)
            out.append(SYM[cls])
            if cls == "swap":
                swaps.append(r)
        else:
            out.append("-")
    return "".join(out), swaps

for label in ("2026", "2024"):
    base, suf = A7.TOMES[label]
    H, W, F, info = A7.load_tome(label, base, suf)
    pats = collections.Counter()
    swapsets = collections.Counter()
    ex = {}
    for m in sorted(H):
        s, sw = sig_for(F.get(m, {}))
        key = s[:26]
        pats[key] += 1
        swapsets[tuple(sw[:4])] += 1
        ex.setdefault(key, m)
    print(f"== {label}: matches {len(H)}; distinct first-26 boundary patterns {len(pats)}")
    print(" top patterns (pattern, count):")
    for p, n in pats.most_common(25):
        print(f"   {p:<28} {n:>5}")
    print(" most common first-4 swap positions:")
    for p, n in swapsets.most_common(15):
        print(f"   {str(p):<24} {n:>5}")
