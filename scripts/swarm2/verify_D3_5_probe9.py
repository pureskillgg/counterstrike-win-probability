"""Verifier D3-5, probe 9: rerun the claimed derivation with the format-appropriate max_rounds
(MR 16 for 2v2, MR 24 for 5v5 and other formats) to measure how much of the claimed disagreement is the schedule.
Read only. Prints to stdout. Reuses verify_D3_5_final.build / formats / swap_expected.
"""
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"G:\pureskill\csds-cs2\scripts\swarm2")))
import verify_D3_5_final as V  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
D = V.build("2024")
FMT = V.formats(D)
MR_OF = lambda m: 16 if FMT.get(m) == "2v2" else 24  # ASSUMPTION: 2v2 = MR16, supported by the 2026 header (max_rounds 16 for 2v2)

by_match = collections.defaultdict(dict)
for (m, r), w in D["A_first"].items():
    by_match[m][r] = w
live = dict(D["FE"])
gap, first_dis, compared, agree = set(), {}, collections.Counter(), collections.Counter()
for m, rounds in by_match.items():
    rs = sorted(rounds)
    if rs != list(range(1, len(rs) + 1)):
        gap.add(m)
        continue
    mr = MR_OF(m)
    side_a = V.T
    a = b = 0
    for r in rs:
        derived = (b - a) if side_a == V.T else (a - b)
        if (m, r) in live:
            lt, lc = live[(m, r)]
            ok = derived == (lc - lt)
            fmt = FMT.get(m, "none")
            compared[fmt] += 1
            agree[fmt] += int(ok)
            if not ok and m not in first_dis:
                first_dis[m] = (r, fmt)
        w = rounds[r]
        if w == side_a:
            a += 1
        elif w == (V.CT if side_a == V.T else V.T):
            b += 1
        if V.swap_expected(r, mr):
            side_a = V.CT if side_a == V.T else V.T

print("non-gap matches with lead comparisons, by format (compared, agree):",
      {k: (compared[k], agree[k]) for k in compared if k in ("5v5", "2v2") or compared[k] > 0})
dis_fmt = collections.Counter(fmt for (r, fmt) in first_dis.values())
print("matches with any disagreement, by format (MR16 for 2v2):", dict(dis_fmt))
print("first disagreeing round by format:", sorted(collections.Counter((fmt, r) for r, fmt in first_dis.values()).items()))
clean5 = [m for m in by_match if V.day_of(m) >= 12 and FMT.get(m) == "5v5" and m not in gap]
print("5v5 day 12-25 non-gap matches:", len(clean5), "| with any disagreement:", sum(1 for m in clean5 if m in first_dis))
print("  first disagreeing round:", sorted(collections.Counter(first_dis[m][0] for m in clean5 if m in first_dis).items()))
d01 = [m for m in by_match if V.day_of(m) < 12 and m not in gap]
print("day 01-07 non-gap matches (round labels start at 1):", len(d01))
