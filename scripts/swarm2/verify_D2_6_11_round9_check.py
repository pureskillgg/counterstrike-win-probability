"""verify D2-6: are the 10 round-9-only flips in matches that also flip at round 13 (flips across all rounds)?"""
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
MK = "match_key"
spec = importlib.util.spec_from_file_location("v08", ROOT / "scripts" / "swarm2" / "verify_D2_6_07_sx_validation.py")
v07 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v07)
v03 = v07.v03
import polars as pl

rt, panel, hd, info = v03.build("2024")
by = defaultdict(dict)
for r in rt.to_dicts():
    by[r[MK]][r["round"]] = r

for mk, byr in by.items():
    for rd, r in byr.items():
        r['W_sc'] = None
        sx, sxn = r['sx'], r['sx_next']
        if sx in (2, 3) and sxn in (2, 3) and None not in (r['t0'], r['ct0'], r['t1'], r['ct1']):
            x0 = r['t0'] if sx == 2 else r['ct0']; y0 = r['ct0'] if sx == 2 else r['t0']
            x1 = r['t1'] if sxn == 2 else r['ct1']; y1 = r['ct1'] if sxn == 2 else r['t1']
            dX, dY = x1 - x0, y1 - y0
            if dX == 1 and dY == 0: r['W_sc'] = sx
            elif dY == 1 and dX == 0: r['W_sc'] = 5 - sx

targets = [
 "csds/2024/02/13/d60d9e55-41d7-4c3d-aeb2-eb4bd4e0ac37/csds",
 "csds/2024/02/18/c23bebbf-7879-46aa-852b-6ec95e823708/csds",
 "csds/2024/02/18/e61be8fb-0246-435b-9674-c2db6e9abc7d/csds",
 "csds/2024/02/19/b5598ffe-6f91-4dc4-bf15-293206dd44e7/csds",
 "csds/2024/02/20/2c3ad4c1-deab-49cd-82a0-8b92d6da39ac/csds",
 "csds/2024/02/20/5455b265-c5ea-4d38-a05f-586d34fd3b4f/csds",
 "csds/2024/02/21/ab8c0b58-bd6a-4b5b-abe6-394a1dd46717/csds",
 "csds/2024/02/23/19e97c7d-17d5-4078-b2e7-50c13f014287/csds",
 "csds/2024/02/24/ac4a8425-f20d-4444-8632-b58bb42f67d5/csds",
 "csds/2024/02/25/662381a2-b8a2-4d3d-8f81-d91e04fec788/csds",
]
for mk in targets:
    byr = by[mk]
    rs = sorted(byr)
    sx1 = byr[1]["sx"]
    flips = []
    for rd in rs:
        r = byr[rd]
        T = r["W_sc"] if r["W_sc"] is not None else (r["bomb_side"] if r["bomb_side"] is not None else r["elim_side"])
        if T is not None and r["R"] != T:
            flips.append((rd, "score" if r["W_sc"] is not None else "bomb/elim"))
    print(mk.split("/")[3], mk.split("/")[4][:8], "rmax", max(rs), "round1 sx", sx1, "round1 nX", byr[1]["nX"], "flips", flips)
