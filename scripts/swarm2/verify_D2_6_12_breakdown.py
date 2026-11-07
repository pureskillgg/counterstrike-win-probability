"""verify D2-6: breakdown of the flagged contiguous 2024 matches that the lead chain does not flag (33)."""
import importlib.util
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
MK = "match_key"
spec = importlib.util.spec_from_file_location("v07", ROOT / "scripts" / "swarm2" / "verify_D2_6_07_sx_validation.py")
v07 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v07)
v03 = v07.v03

rt, panel, hd, info = v03.build("2024")
by = defaultdict(dict)
for r in rt.to_dicts():
    by[r[MK]][r["round"]] = r
for mk, byr in by.items():
    for rd, r in byr.items():
        r["sx"] = r["sx"] if r["sx"] in (2, 3) else None
        r["W_sc"] = None
        sx, sxn = r["sx"], r["sx_next"]
        if sx in (2, 3) and sxn in (2, 3) and None not in (r["t0"], r["ct0"], r["t1"], r["ct1"]):
            x0 = r["t0"] if sx == 2 else r["ct0"]; y0 = r["ct0"] if sx == 2 else r["t0"]
            x1 = r["t1"] if sxn == 2 else r["ct1"]; y1 = r["ct1"] if sxn == 2 else r["t1"]
            dX, dY = x1 - x0, y1 - y0
            if dX == 1 and dY == 0: r["W_sc"] = sx
            elif dY == 1 and dX == 0: r["W_sc"] = 5 - sx
        r["T"] = r["W_sc"] if r["W_sc"] is not None else (r["bomb_side"] if r["bomb_side"] is not None else r["elim_side"])

cls = Counter()
rows = []
for mk, byr in by.items():
    rs = sorted(byr)
    if not (rs[0] == 1 and rs[-1] == len(rs)):
        continue
    rmax = rs[-1]
    flips = [rd for rd in rs if byr[rd]["T"] is not None and byr[rd]["R"] != byr[rd]["T"]]
    if not flips:
        continue
    # lead chain (raw sx, break at first null)
    sX = sY = 0; fb = None; stop = None
    for rd in range(1, rmax + 1):
        r = byr[rd]
        if r["sx"] is None:
            stop = rd; break
        if r["t0"] is not None and r["ct0"] is not None:
            rs_lead = r["ct0"] - r["t0"]
            tr = (sX - sY) if r["sx"] == 3 else (sY - sX)
            if tr != rs_lead and fb is None:
                fb = rd
        sX += 1 if r["R"] == r["sx"] else 0
        sY += 0 if r["R"] == r["sx"] else 1
    if fb is not None:
        cls["divergent"] += 1
        continue
    f = flips[0]
    tag = f"first flip {f}, rmax {rmax}, chain stop {stop}"
    if f == rmax:
        cls["flip only in last round (rmax=%d)" % rmax] += 1
    elif stop is not None and stop <= f + 1:
        cls["masked by null sx before flip+1 (flip round %d)" % f] += 1
    else:
        cls["other"] += 1
    rows.append((mk, tag, flips))
for k, v in sorted(cls.items()):
    print(f"{v:>4}  {k}")
print("details of non-divergent flagged matches:")
for x in rows:
    print("  ", x[1], "flips", x[2])
