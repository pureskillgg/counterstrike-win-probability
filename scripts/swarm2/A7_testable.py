"""A7: how many matches can test side vs team attachment at all?

Proof of side attachment: an observed crossover (swap-only boundary) can only come from side-attached
scores, since team-attached scores never cross. Counts matches with at least one swap-only boundary.
Also counts matches with at least one unambiguous scheduled-swap boundary (the ATT test's domain).
"""
import importlib.util
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
spec = importlib.util.spec_from_file_location("A7", ROOT / "scripts" / "swarm2" / "A7_score_coverage.py")
A7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A7)
RULE = A7.half_rule(*A7.TOMES["2024"])
for label in ("2026", "2024"):
    base, suf = A7.TOMES[label]
    H, W, F, info = A7.load_tome(label, base, suf)
    n = len(H)
    n_cross = 0
    n_sched_testable = 0
    n_sched_testable_kept = 0
    n_nobound = 0
    for m in H:
        Fm = F.get(m, {})
        mr = int(H[m]["max_rounds"]) if label == "2026" else A7.mr_best(Fm, A7.MR24_CANDIDATES, prefer=RULE.get(m, 24))[0]
        crossed = False
        sched_ok = False
        for r in sorted(Fm):
            if r + 1 not in Fm:
                continue
            t, c = Fm[r]
            t2, c2 = Fm[r + 1]
            cls = A7.boundary_class(t, c, t2, c2)
            if cls == "swap":
                crossed = True
            if A7.swap_expected(r, mr) and cls in ("swap", "noswap"):
                sched_ok = True
        n_cross += int(crossed)
        n_sched_testable += int(sched_ok)
        n_nobound += int(not any(r + 1 in Fm for r in Fm))
    print(f"== {label}: matches {n}; with an observed crossover (proves side attachment): {n_cross} "
          f"({100 * n_cross / n:.1f}%); with >=1 unambiguous scheduled-swap boundary: {n_sched_testable}; "
          f"no consecutive freeze pair: {n_nobound}")
