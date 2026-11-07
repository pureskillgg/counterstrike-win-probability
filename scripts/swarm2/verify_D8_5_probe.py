"""verify_D8_5_probe: print per-round sequences for example (8,) other_misfit matches (D8-5 verifier).

Reads the scratchpad rounds.parquet written by verify_D8_5_build.py. Prints to stdout only.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D8_5_probe.py
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
OUT = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D8_5")
T, CT = 2, 3

R = pl.read_parquet(OUT / "rounds.parquet").sort(["match_key", "round"])
MAT = {}
for x in R.iter_rows(named=True):
    m = MAT.setdefault(x["match_key"], {"map": x["map_name"], "tA": int(x["tA"]), "tB": int(x["tB"]), "rd": []})
    m["rd"].append(x)


def step(x, y):
    if None in (x["tb"], x["cb"], y["tb"], y["cb"]):
        return "missing"
    tT = 1 if x["winner"] == T else 0
    tC = 1 - tT
    ns = (y["tb"] == x["tb"] + tT) and (y["cb"] == x["cb"] + tC)
    sw = (y["tb"] == x["cb"] + tC) and (y["cb"] == x["tb"] + tT)
    return "both" if (ns and sw) else "noswap" if ns else "swap" if sw else "ANOMALY"


def show(key, m):
    rd = m["rd"]
    n = len(rd)
    print(f"\n{key}  map={m['map']}  n={n}  header tA={m['tA']} tB={m['tB']}  (complete={n == m['tA'] + m['tB']})")
    print(f"{'r':>3}{'winner':>8}{'tb_fe':>7}{'cb_fe':>7}  step(r->r+1)  bside")
    for i, x in enumerate(rd):
        st = step(x, rd[i + 1]) if i + 1 < n and rd[i + 1]["round"] == x["round"] + 1 else "-"
        w = "T" if x["winner"] == T else "CT"
        print(f"{x['round']:>3}{w:>8}{str(x['tb']):>7}{str(x['cb']):>7}  {st:<12}  "
              f"{'' if x['bside'] is None else ('CT' if x['bside'] == CT else 'T')}")


# example (8,) matches: first 3 with n=16 complete, first 2 with n=24 (non-16) etc.
def sig8(m):
    rd = m["rd"]
    live = {x["round"]: (x["tb"], x["cb"]) for x in rd if x["tb"] is not None}
    sw = []
    for r in sorted(live):
        if r + 1 not in live:
            continue
        t1, c1 = live[r]
        t2, c2 = live[r + 1]
        dt, dc = t2 - t1, c2 - c1
        if (dc == 1 and dt == 0) or (dt == 1 and dc == 0) or (dt == 0 and dc == 0):
            continue
        if dt * dc <= 0:
            sw.append(r)
    return tuple(sw)


cands = [(k, m) for k, m in MAT.items() if sig8(m) == (8,) and len(m["rd"]) == m["tA"] + m["tB"]]
print(f"(8,) complete matches: {len(cands)}")
for k, m in cands[:3]:
    show(k, m)
ex = [(k, m) for k, m in MAT.items() if sig8(m) == (12,) and len(m["rd"]) == m["tA"] + m["tB"]]
print(f"\n(12,) complete matches for comparison: {len(ex)}")
for k, m in ex[:1]:
    show(k, m)
print("\nPROBE DONE")
