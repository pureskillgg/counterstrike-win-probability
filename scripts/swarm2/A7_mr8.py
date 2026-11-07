"""A7: are the swap-at-round-8 matches MR16 (halftime after round 8)?

For every match in both tomes: the score-only boundary string (rounds 1..max), the swap-only
positions, the header max_rounds (2026 only, a real column) and the header final score sum.
Compares, for the matches whose only swap-only boundary is round 8, the header max_rounds (2026)
and the total rounds played (round_end count), and reports which schedule (MR16, MR24, MR30) the
boundary string fits best. No winner codes are used for the fit.
"""
import collections
import importlib.util
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
spec = importlib.util.spec_from_file_location("A7", ROOT / "scripts" / "swarm2" / "A7_score_coverage.py")
A7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A7)
RULE = A7.half_rule(*A7.TOMES["2024"])

SYM = {"noswap": "n", "swap": "s", "ambig": "a", "neither": "x"}
CANDS = (16, 24, 30)


def boundary_syms(Fm):
    rs = sorted(Fm)
    if not rs:
        return "", []
    mx = rs[-1]
    out, swaps = [], []
    for r in range(1, mx):
        if r in Fm and r + 1 in Fm:
            t, c = Fm[r]
            t2, c2 = Fm[r + 1]
            cls = A7.boundary_class(t, c, t2, c2)
            out.append(SYM[cls])
            if cls == "swap":
                swaps.append(r)
        else:
            out.append("-")
    return "".join(out), swaps


def fit_bad(Fm, mr):
    bad = 0
    for r in Fm:
        if r + 1 not in Fm:
            continue
        t, c = Fm[r]
        t2, c2 = Fm[r + 1]
        cls = A7.boundary_class(t, c, t2, c2)
        if cls in ("ambig", "neither"):
            continue
        expect = "swap" if A7.swap_expected(r, mr) else "noswap"
        bad += int(cls != expect)
    return bad


for label in ("2026", "2024"):
    base, suf = A7.TOMES[label]
    H, W, F, info = A7.load_tome(label, base, suf)
    print(f"== {label}")
    rows = []
    for m in sorted(H):
        Fm, Wm = F.get(m, {}), W.get(m, {})
        s, sw = boundary_syms(Fm)
        n_rounds = len(Wm)
        mr_hdr = H[m].get("max_rounds")
        fsum = H[m]["t_starters_score_final"] + H[m]["ct_starters_score_final"]
        fit = {k: fit_bad(Fm, k) for k in CANDS}
        rows.append((m, tuple(sw[:3]), s, n_rounds, mr_hdr, fsum, fit))
    only8 = [r for r in rows if r[1] == (8,)]
    print(f"  matches with swap-only boundaries exactly at (8,) first: {len(only8)}")
    if label == "2026":
        print("  header max_rounds for the (8,) matches:", dict(collections.Counter(r[4] for r in only8)))
        print("  header max_rounds for all matches:", dict(collections.Counter(r[4] for r in rows)))
    else:
        print("  2024 (8,) matches: round_end counts (top):",
              collections.Counter(r[3] for r in only8).most_common(8))
        print("  2024 header final sum for (8,) matches (top):",
              collections.Counter(r[5] for r in only8).most_common(8))
    print("  (8,) matches: best fit among candidates (count of matches by argmin, ties listed):")
    tally = collections.Counter()
    for r in only8:
        best = min(r[6].values())
        tally[tuple(k for k in CANDS if r[6][k] == best)] += 1
    print("    ", dict(tally))
    print("  (8,) matches: sample patterns (first 40 chars):")
    for r in only8[:4]:
        print("    ", r[0][-40:], r[2][:40], "rounds_end", r[3], "hdr_mr", r[4], "fsum", r[5], "fit", r[6])
    print("  (12,) matches: round_end counts (top):",
          collections.Counter(r[3] for r in rows if r[1] == (12,)).most_common(6))
    print("  (12,) matches: header max_rounds:",
          dict(collections.Counter(r[4] for r in rows if r[1] == (12,))) if label == "2026" else "n/a")
    print("  matches with no swap-only boundary:",
          sum(1 for r in rows if r[1] == ()), "; header max_rounds:",
          dict(collections.Counter(r[4] for r in rows if r[1] == ())) if label == "2026" else "n/a")
    if label == "2026":
        print("  2026: max_rounds vs swap position (first swap):")
        ct = collections.Counter((r[4], r[1][0] if r[1] else None) for r in rows)
        for k, v in sorted(ct.items(), key=lambda x: (str(x[0][0]), str(x[0][1]))):
            print("    mr", k[0], "first swap", k[1], "count", v)
