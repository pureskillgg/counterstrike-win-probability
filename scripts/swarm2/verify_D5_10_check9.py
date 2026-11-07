"""verify D5_10, step 4: two side checks.

(a) The 8->9 side-swap group (excluded from the claim): bomb truth at rounds 8-10 and the round-9 class.
(b) 2026 MR24 matches: round-label coverage (contiguous? any labels above 24?), and the matches where the
    D5 hard-coded schedule disagrees with the bake-off schedule.
Read-only; uses verify_D5_10_analysis.py functions (imported, its main() not run).
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_10_check9.py
"""
import collections
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import verify_D5_10_analysis as A  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def main():
    R24, M24 = A.load("2024")
    s8 = {m: A.swap8_flag(R24.get(m, {})) for m in M24}

    print("== (a) 8->9 swap group, regime A and B: bomb truth at rounds 7-10 ==")
    for reg in ["A", "B"]:
        tab = collections.defaultdict(lambda: [0, 0])
        for m, rr in R24.items():
            if A.regime_of(M24[m]) != reg or not s8[m]:
                continue
            for r in range(7, 11):
                x = rr.get(r)
                if not x:
                    continue
                if x["n_planted"] >= 1 and (x["n_exploded"] + x["n_defused"]) == 1 and x["winner"] is not None:
                    truth = A.T if x["n_exploded"] else A.CT
                    tab[r][0] += 1
                    tab[r][1] += int(truth != x["winner"])
        print(f"  regime {reg}, swap8 matches (round: bomb rounds/mismatches): "
              + ", ".join(f"{r}:{v[0]}/{v[1]}" for r, v in sorted(tab.items())))
    cl = collections.Counter()
    for m, rr in R24.items():
        if A.regime_of(M24[m]) == "A" and s8[m]:
            cl[str(A.hits_at(rr, 9))] += 1
    print("  regime A swap8 matches, boundary-9 classes:", dict(cl))

    print("\n== (b) 2026 MR24 round-label coverage ==")
    R26, M26 = A.load("2026")
    mr24 = [m for m in M26 if M26[m]["max_rounds"] == 24]
    lab_max = collections.Counter()
    gaps = 0
    above24 = 0
    odd = []
    for m in mr24:
        rs = sorted(R26[m])
        if rs != list(range(1, len(rs) + 1)):
            gaps += 1
        if rs and rs[-1] > 24:
            above24 += 1
            if len(odd) < 5:
                odd.append((m[-10:], rs[-1], len(rs), M26[m]["t_starters_score_final"], M26[m]["ct_starters_score_final"]))
        lab_max[min(rs[-1], 40) if rs else 0] += 1
    print(f"  2026 MR24 matches {len(mr24)}; label sets not contiguous 1..n: {gaps}; matches with labels above 24: {above24}")
    print("  last-label histogram (cap 40):", sorted(lab_max.items())[:45])
    print("  examples with labels above 24 (key tail, last label, count, header t, header ct):", odd)
    d_bo = sum(1 for m in mr24 if A.states_for(R26, M26, A.SCHEDS["BO_MR24"], m, "winner"))
    d_d5 = sum(1 for m in mr24 if A.states_for(R26, M26, A.SCHEDS["D5_12_24_27"], m, "winner"))
    print(f"  matches with any state under BO map: {d_bo}; under D5 map: {d_d5}")
    x_bo = sum(1 for m in mr24 if any(d != 0 for _, d in A.states_for(R26, M26, A.SCHEDS["BO_MR24"], m, "winner")))
    x_d5 = sum(1 for m in mr24 if any(d != 0 for _, d in A.states_for(R26, M26, A.SCHEDS["D5_12_24_27"], m, "winner")))
    print(f"  matches with a nonzero diff: BO map {x_bo}; D5 map {x_d5}")


if __name__ == "__main__":
    main()
