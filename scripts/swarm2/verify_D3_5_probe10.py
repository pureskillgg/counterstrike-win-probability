"""Verifier D3-5, probe 10: reconcile the 5v5 round-14 first disagreements (derived vs live) with the round-13
flip rate. For each 5v5 day 12-25 non-gap match: is round 13's round_end winner consistent with the score-field
transition 13 -> 14 (unique or ambiguous), and does the cumulative derived lead match the live lead at 14?
Read only. Prints to stdout.
"""
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"G:\pureskill\csds-cs2\scripts\swarm2")))
import verify_D3_5_final as V  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
D = V.build("2024")
FMT = V.formats(D)
by_match = collections.defaultdict(dict)
for (m, r), w in D["A_first"].items():
    by_match[m][r] = w
cats = collections.Counter()
ex = collections.defaultdict(list)
for m, rounds in by_match.items():
    if V.day_of(m) < 12 or FMT.get(m) != "5v5":
        continue
    rs = sorted(rounds)
    if rs != list(range(1, len(rs) + 1)):
        continue
    if 13 not in rounds or (m, 13) not in D["FE"] or (m, 14) not in D["FE"]:
        cats["no freeze 13 or 14"] += 1
        continue
    # cumulative derived lead through round 13 (team tally, MR24 schedule as claimed)
    side_a = V.T
    a = b = 0
    derived14 = None
    for r in rs:
        if r == 14:
            break
        w = rounds[r]
        if w == side_a:
            a += 1
        elif w == (V.CT if side_a == V.T else V.T):
            b += 1
        if V.swap_expected(r, 24):
            side_a = V.CT if side_a == V.T else V.T
    # the lead before round 14 is the score difference from the team tally with the final side_a
    derived14 = (b - a) if side_a == V.T else (a - b)
    lt, lc = D["FE"][(m, 14)]
    live14 = lc - lt
    ok14 = derived14 == live14
    t13, c13 = D["FE"][(m, 13)]
    t14, c14 = D["FE"][(m, 14)]
    wset = V.classify_winners(t13, c13, t14, c14)
    a13 = rounds[13]
    if len(wset) == 1:
        c_status = "unique_" + ("agrees" if a13 in wset else "disagrees")
    elif len(wset) == 0:
        c_status = "no_pattern"
    else:
        c_status = "ambiguous_" + ("A_in_set" if a13 in wset else "A_not_in_set")
    key = (ok14, c_status)
    cats[key] += 1
    if len(ex[key]) < 2:
        ex[key].append((m[-12:], (t13, c13), (t14, c14), a13, sorted(wset), derived14, live14))
print("5v5 day12-25 non-gap matches, by (derived14 == live14, round-13 score pattern vs A13):")
for k, v in sorted(cats.items(), key=lambda kv: str(kv[0])):
    print("  ", k, v)
print("examples (match, freeze13, freeze14, A13, candidate winners, derived14, live14):")
for k, es in ex.items():
    for e in es:
        print("  ", k, e)
