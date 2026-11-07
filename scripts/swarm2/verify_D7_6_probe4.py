"""verify D7-6 probe4: player_count composition of DP swap-after-8 matches (pre and post), and duplicate-row check."""
import sys
import collections

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"G:\pureskill\csds-cs2\scripts\swarm2")
import verify_D7_6_common as C  # noqa: E402

D = C.load()
PC = collections.defaultdict(set)
for (k, r), v in D["RE"].items():
    PC[k].add(v["player_count"])
DPS = collections.defaultdict(list)
for (k, r), v in D["LEDGER"].items():
    if v[1]:
        DPS[k].append(r)
for per in ["pre", "post"]:
    ks8 = [k for k in DPS if 8 in DPS[k] and ((D["MDT"][k] < C.ONSET) == (per == "pre"))]
    print(per, "DP swap after 8:", len(ks8), "| player_count sets:", collections.Counter(tuple(sorted(PC[k])) for k in ks8).most_common(6))
    ks12 = [k for k in DPS if 12 in DPS[k] and ((D["MDT"][k] < C.ONSET) == (per == "pre"))]
    print(per, "DP swap after 12:", len(ks12), "| player_count sets:", collections.Counter(tuple(sorted(PC[k])) for k in ks12).most_common(4))
print("matches in tome:", len(D["MDT"]), "| matches with any round_end:", len(PC))
