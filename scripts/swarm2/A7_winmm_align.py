"""A7: where do the 2024 recorded-winner flips (WINMM) sit relative to the score-based swap schedule?

For every 2024 match: the chosen max_rounds (score-only fit), the score-only swap-only positions
(boundaries classified as swap), and the WINMM rounds. Tests whether each WINMM round is the first
round after a swap (swap boundary at round-1), and reports the flip rate by chosen max_rounds.
Also checks the 2026 kept set and excluded matches for the same alignment (expected: no WINMM).
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
T, CT = A7.T, A7.CT

base, suf = A7.TOMES["2024"]
H, W, F, info = A7.load_tome("2024", base, suf)

by_mr = collections.Counter()
mr_matches = collections.Counter()
align = collections.Counter()
win_round_after_swap = collections.Counter()
mr9_examples = []
for m in sorted(H):
    Fm, Wm = F.get(m, {}), W.get(m, {})
    best, bad, ties, fit = A7.mr_best(Fm, A7.MR24_CANDIDATES, prefer=RULE.get(m, 24))
    mr_matches[best] += 1
    res, det = A7.analyze_match(m, best, Wm, Fm)
    swaps = []
    rs = sorted(Fm)
    for r in rs:
        if r + 1 in Fm:
            t, c = Fm[r]
            t2, c2 = Fm[r + 1]
            if A7.boundary_class(t, c, t2, c2) == "swap":
                swaps.append(r)
    # expected swap positions under the chosen schedule (up to the last round seen)
    last = rs[-1] if rs else 0
    sched = [r for r in range(1, last) if A7.swap_expected(r, best)]
    wm = [int(x) for x in res.get("winmm_list", "").split(",") if x]
    for r in wm:
        by_mr[best] += 1
        # is the WINMM round the first round after a scheduled swap?
        after = (r - 1) in sched
        align["WINMM_after_sched_swap" if after else "WINMM_not_after_sched_swap"] += 1
        win_round_after_swap[(best, r)] += 1
    if best == 9 and len(mr9_examples) < 4:
        mr9_examples.append((m[-40:], best, bad, ties, swaps[:4], sched[:4], last, len(Wm), wm[:4],
                             res.get("first_dis_round", 0)))

print(f"2024 chosen max_rounds counts (top 12): {mr_matches.most_common(12)}")
print(f"2024 WINMM rounds by chosen mr: {dict(by_mr)}")
print(f"2024 WINMM alignment: {dict(align)}")
print(f"2024 WINMM (mr, round) top 15: {win_round_after_swap.most_common(15)}")

print("\nexamples chosen mr=9: (match tail, mr, fit_bad, ties, score swaps, sched, last_round, n_winners, WINMM, first_DIS)")
for e in mr9_examples:
    print("  ", e)

# mr=9 matches: how many have any score swap at all, and their round counts
mr9 = [m for m in H if A7.mr_best(F.get(m, {}), A7.MR24_CANDIDATES, prefer=RULE.get(m, 24))[0] == 9]
sw9 = collections.Counter()
rc9 = collections.Counter()
for m in mr9:
    Fm = F.get(m, {})
    rs = sorted(Fm)
    sw = []
    for r in rs:
        if r + 1 in Fm:
            t, c = Fm[r]
            t2, c2 = Fm[r + 1]
            if A7.boundary_class(t, c, t2, c2) == "swap":
                sw.append(r)
    sw9[tuple(sw[:3])] += 1
    rc9[len(W.get(m, {}))] += 1
print(f"\nchosen mr=9 matches: {len(mr9)}; swap-only positions (top): {sw9.most_common(8)}")
print(f"chosen mr=9 round_end counts (top): {rc9.most_common(10)}")

# 2026 alignment check using the header max_rounds and the same score-only test
base26, suf26 = A7.TOMES["2026"]
H26, W26, F26, info26 = A7.load_tome("2026", base26, suf26)
w26 = 0
for m in H26:
    mr = int(H26[m]["max_rounds"])
    res, _ = A7.analyze_match(m, mr, W26.get(m, {}), F26.get(m, {}))
    w26 += res.get("winmm_rounds", 0)
print(f"\n2026 WINMM total with header max_rounds: {w26}")
