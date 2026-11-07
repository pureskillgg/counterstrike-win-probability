"""verify D5_10, step 6: the 51 regime-A matches with an early team-A diff (label 10) and the 8->9 rule.

(a) Boundary-8 class for the early-onset matches (ambiguous N/S means the claim's swap8 rule, sw AND NOT ns,
    cannot flag them).
(b) Header final max score: 9 means a completed MR16 match (first to 9); 13 means MR24.
(c) Header test for regime-A no-swap8 complete matches with round 13 flipped, and with 13/25/28 flipped.
Read-only; uses verify_D5_10_analysis.py functions.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_10_check_early.py
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
    regA_ns = lambda m: A.regime_of(M24[m]) == "A" and not s8[m]

    early = []
    for m in M24:
        if not regA_ns(m):
            continue
        st = A.states_for(R24, M24, A.SCHEDS["D5_12_24_27"], m, "winner")
        nz = [r for r, d in st if d != 0]
        if nz and min(nz) < 14:
            early.append(m)
    print(f"early-onset matches (regime A, no swap8, first diff < 14): {len(early)}")

    b8 = collections.Counter(str(A.hits_at(R24[m], 8)) for m in early)
    print("  boundary-8 classes:", dict(b8))
    b9 = collections.Counter(str(A.hits_at(R24[m], 9)) for m in early)
    print("  boundary-9 classes:", dict(b9))
    mx = collections.Counter(max(M24[m]["t_starters_score_final"], M24[m]["ct_starters_score_final"]) for m in early)
    print("  header max final score (9 = completed MR16, 13 = completed MR24):", dict(mx))
    print("  header sums:", dict(collections.Counter(M24[m]["t_starters_score_final"] + M24[m]["ct_starters_score_final"] for m in early)))

    # Would the claim's swap8 rule ever flag an ambiguous 8->9 boundary? Count regime-A matches by 8 class.
    all8 = collections.Counter()
    for m in M24:
        if A.regime_of(M24[m]) != "A":
            continue
        all8[(str(A.hits_at(R24[m], 8)), s8[m])] += 1
    print("\nregime A boundary-8 class x claim swap8 flag (all 2478 A matches):")
    for k, v in sorted(all8.items()):
        print(f"  class {k[0]:<22} swap8={k[1]!s:<5} {v}")

    print("\nheader max score among regime-A no-swap8 matches:",
          dict(collections.Counter(max(M24[m]['t_starters_score_final'], M24[m]['ct_starters_score_final'])
                                   for m in M24 if regA_ns(m))))

    print("\nheader test, regime A no-swap8, complete matches:")
    A.header_test(R24, M24, lambda mat: A.SCHEDS["BO_MR24"], regA_ns, "no flip")
    A.header_test(R24, M24, lambda mat: A.SCHEDS["BO_MR24"], regA_ns, "flip 13", flip_rounds=frozenset({13}))
    A.header_test(R24, M24, lambda mat: A.SCHEDS["BO_MR24"], regA_ns, "flip 13, 25, 28",
                  flip_rounds=frozenset({13, 25, 28}))


if __name__ == "__main__":
    main()
