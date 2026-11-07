"""verify D2-6: round-13 R vs sx-free bomb/elimination truth, split by contiguity, with denominators; day split of contiguity."""
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

rt, panel, hd, info = v03.build("2024")
by = defaultdict(dict)
for r in rt.to_dicts():
    by[r[MK]][r["round"]] = r
c = Counter()
day_c = Counter()
day_n = Counter()
for mk, byr in by.items():
    rs = sorted(byr)
    contig = rs[0] == 1 and rs[-1] == len(rs)
    day = mk.split("/")[2] + "/" + mk.split("/")[3]
    day_n[(day, contig)] += 1
    r13 = byr.get(13)
    if r13 is None:
        c[("contig" if contig else "noncontig", "no round 13 row")] += 1
        continue
    ind = r13["bomb_side"] if r13["bomb_side"] is not None else r13["elim_side"]
    key = "contig" if contig else "noncontig"
    if ind is None:
        c[(key, "no bomb/elim truth")] += 1
        continue
    c[(key, "known")] += 1
    if r13["R"] != ind:
        c[(key, "R!=truth")] += 1
        day_c[day] += 1
for k in sorted(c):
    print(k, c[k])
print("contiguous round-13 R!=truth rate:", c[("contig", "R!=truth")], "/", c[("contig", "known")],
      "=", round(c[("contig", "R!=truth")] / max(c[("contig", "known")], 1), 4))
print("non-contiguous round-13 R!=truth rate:", c[("noncontig", "R!=truth")], "/", c[("noncontig", "known")])
# day split: contiguous vs non-contiguous match counts for a few days
for d in ["02/01", "02/02", "02/05", "02/07", "02/08", "02/12", "02/13", "02/17", "02/20", "02/25"]:
    print(d, "contig", day_n[(d, True)], "noncontig", day_n[(d, False)])
