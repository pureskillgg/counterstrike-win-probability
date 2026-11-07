"""verify D2-6: scope check. Matches excluded by the contiguity rule (round_end must list rounds 1..max).
Counts round-13 flips (R vs bomb, else elimination; both sx-free) inside excluded matches, so the
affected-match count can be stated with its scope. Also reports how many rounds the gaps hide.

Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_D2_6_09_scope.py
"""
import importlib.util
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
MK = "match_key"

spec = importlib.util.spec_from_file_location("v03", ROOT / "scripts" / "swarm2" / "verify_D2_6_03_main.py")
v03 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v03)


def main():
    for label in ["2024", "2026"]:
        rt, panel, hd, info = v03.build(label)
        by = defaultdict(dict)
        for r in rt.to_dicts():
            by[r[MK]][r["round"]] = r
        cnt = Counter()
        gap_sizes = Counter()
        flip13_noncontig = 0
        flip13_contig = 0
        flagged_noncontig = 0
        n13_known_noncontig = 0
        for mk, byr in by.items():
            rs = sorted(byr)
            contig = rs[0] == 1 and rs[-1] == len(rs)
            cnt["contiguous" if contig else "non-contiguous"] += 1
            if not contig:
                missing = rs[-1] - len(rs)
                gap_sizes[min(missing, 5)] += 1
            r13 = byr.get(13)
            ind = None
            if r13 is not None:
                ind = r13["bomb_side"] if r13["bomb_side"] is not None else r13["elim_side"]
            if ind is not None:
                if contig:
                    flip13_contig += int(r13["R"] != ind)
                else:
                    n13_known_noncontig += 1
                    flip13_noncontig += int(r13["R"] != ind)
            if not contig:
                if any(byr[k]["bomb_side"] is not None and byr[k]["R"] != byr[k]["bomb_side"] for k in rs if k in byr):
                    flagged_noncontig += 1
        print(f"\n######## {label} ########")
        print(f"  matches in round_end: {len(by)}; {dict(cnt)}")
        print(f"  missing round numbers in non-contiguous matches (min(gap,5): matches): {sorted(gap_sizes.items())}")
        print(f"  round 13 with bomb/elim truth: contiguous R!=truth {flip13_contig}; non-contiguous R!=truth {flip13_noncontig} "
              f"of {n13_known_noncontig} known")
        print(f"  non-contiguous matches with any round where R != bomb outcome: {flagged_noncontig}")


if __name__ == "__main__":
    main()
