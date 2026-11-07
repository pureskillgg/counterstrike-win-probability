"""verify D5_10, step 7: which source agrees with the physical bomb outcome at the suspect rounds?

For each round with one planted bomb and exactly one resolution (explode -> T won, defuse -> CT won), compare:
  - the round_end winner (label), and
  - the freeze-end winner implied by the boundary class (N -> label; Ninv -> flipped label; ambiguous and other
    classes are counted separately and not used).
ASSUMPTION: the boundary class uses slot semantics (round_state t_score / ct_score follow the side); this is
the same assumption the claim's D5_10 makes for its freeze-end team-A score.
Read-only; uses verify_D5_10_analysis.py functions.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_10_freeze_vs_bomb.py
"""
import collections
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import verify_D5_10_analysis as A  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
ROUNDS = [9, 13, 25, 28]


def main():
    R24, M24 = A.load("2024")
    s8 = {m: A.swap8_flag(R24.get(m, {})) for m in M24}
    tab = collections.Counter()
    for m, rr in R24.items():
        reg = A.regime_of(M24[m])
        grp = "A swap8" if (reg == "A" and s8[m]) else ("A no-swap8" if reg == "A" else "B")
        for r in ROUNDS + [12, 14]:
            x = rr.get(r)
            if not x or x["winner"] is None:
                continue
            if not (x["n_planted"] >= 1 and (x["n_exploded"] + x["n_defused"]) == 1):
                continue
            truth = A.T if x["n_exploded"] else A.CT
            w = x["winner"]
            h = A.hits_at(rr, r)
            if isinstance(h, str):
                cls = h
                fz = None
            elif h == ("N",):
                cls, fz = "N", w
            elif h == ("Ninv",):
                cls, fz = "Ninv", (A.CT if w == A.T else A.T)
            else:
                cls, fz = "amb:" + "/".join(h), None
            tab[(grp, r, "label_vs_bomb", "agree" if w == truth else "disagree")] += 1
            if fz is not None:
                tab[(grp, r, "freeze_vs_bomb", "agree" if fz == truth else "disagree")] += 1
            tab[(grp, r, "class", cls)] += 1
    for k in sorted(tab, key=lambda k: (k[0], k[1], k[2], k[3])):
        print(f"  {k[0]:<12} round {k[1]:>2} {k[2]:<16} {k[3]:<22} {tab[k]}")


if __name__ == "__main__":
    main()
