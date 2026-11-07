"""A7: round-sequence gaps and the DIS-without-WINMM matches (2024 and 2026).

1. Gaps: rounds 1..max with neither a round_end nor a freeze row. Where they sit (first gap round,
   count per match), and whether the match has a winner and freeze row on each side.
2. DIS without WINMM: matches with a disagreement but no unambiguous recorded-winner mismatch.
   Checks whether the boundary before the first disagreement is ambiguous (score tie).
3. Coverage: freeze rows compared by the winner-derived chain versus all freeze rows, per tome.
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

for label in ("2026", "2024"):
    base, suf = A7.TOMES[label]
    H, W, F, info = A7.load_tome(label, base, suf)
    first_gap = collections.Counter()
    n_gap_hist = collections.Counter()
    gap_matches = 0
    gap_both_missing = 0
    freeze_total = 0
    compared_total = 0
    dis_no_win = []
    dis_no_win_first_dis = collections.Counter()
    dis_no_win_amb_before = collections.Counter()
    for m in sorted(H):
        Fm, Wm = F.get(m, {}), W.get(m, {})
        mr = int(H[m]["max_rounds"]) if label == "2026" else A7.mr_best(Fm, A7.MR24_CANDIDATES, prefer=RULE.get(m, 24))[0]
        rounds_all = sorted(set(Wm) | set(Fm))
        mx = rounds_all[-1] if rounds_all else 0
        gaps = [r for r in range(1, mx + 1) if r not in Wm and r not in Fm]
        freeze_total += len(Fm)
        if gaps:
            gap_matches += 1
            first_gap[gaps[0]] += 1
            n_gap_hist[min(len(gaps), 5)] += 1
        res, det = A7.analyze_match(m, mr, Wm, Fm)
        compared_total += res.get("compared_rounds", 0)
        if res.get("dis_rounds", 0) > 0 and res.get("winmm_rounds", 0) == 0:
            dis_no_win.append(m)
            fd = res["first_dis_round"]
            dis_no_win_first_dis[fd] += 1
            # boundary before the first disagreement, and the one before that
            b = fd - 1
            cls = "missing"
            if b in Fm and (b + 1) in Fm:
                t, c = Fm[b]
                t2, c2 = Fm[b + 1]
                cls = A7.boundary_class(t, c, t2, c2)
            dis_no_win_amb_before[cls] += 1
    print(f"== {label}")
    print(f"   matches {len(H)}; matches with a round-sequence gap: {gap_matches}; "
          f"first gap round (top 10): {first_gap.most_common(10)}")
    print(f"   gap length per match (1..5+): {dict(sorted(n_gap_hist.items()))}")
    print(f"   freeze rows total {freeze_total}; freeze rows compared by the winner-derived chain: {compared_total}")
    print(f"   DIS>0 matches with no WINMM: {len(dis_no_win)}; first DIS round: {dis_no_win_first_dis.most_common(6)}")
    print(f"   boundary class just before the first DIS round (no-WINMM matches): {dict(dis_no_win_amb_before)}")
