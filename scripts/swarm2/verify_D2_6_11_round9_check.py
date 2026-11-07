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
 "csds/2024/02/13/m_dbb7b9185c2c999e5a58eba7/csds",
 "csds/2024/02/18/m_c7edff9d03c7d5368ee6549e/csds",
 "csds/2024/02/18/m_8ddf44d7cf84f730f070230f/csds",
 "csds/2024/02/19/m_bcc1f3c593a6990132b80cc5/csds",
 "csds/2024/02/20/m_1b7d27f2deca85693c9c9dcb/csds",
 "csds/2024/02/20/m_b56e73ac0708a0dc588be826/csds",
 "csds/2024/02/21/m_738dd5d4e52428b16372327b/csds",
 "csds/2024/02/23/m_c160948e62202b221ba9706f/csds",
 "csds/2024/02/24/m_0e5f4f1389c78f27b557aab3/csds",
 "csds/2024/02/25/m_df75e0883ba804e2c7402df9/csds",
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
