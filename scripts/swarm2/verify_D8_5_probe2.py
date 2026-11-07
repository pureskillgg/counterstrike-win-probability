"""verify_D8_5_probe2: print per-round sequences for three groups that the model check left open (D8-5 verifier).

Groups: (a) () complete matches whose first clear swap step is at 13 (not 12); (b) () other_misfit matches
(D8 found no swap, census finds a swap at 9); (c) (8,) other_misfit matches where the round-9 flip does not fit.
Prints to stdout only. Reads the scratchpad rounds.parquet written by verify_D8_5_build.py.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D8_5_probe2.py
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
OUT = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D8_5")
T, CT = 2, 3

R = pl.read_parquet(OUT / "rounds.parquet").sort(["match_key", "round"])
MAT = {}
for x in R.iter_rows(named=True):
    m = MAT.setdefault(x["match_key"], {"map": x["map_name"], "tA": int(x["tA"]), "tB": int(x["tB"]),
                                        "day": int(x["day"]), "rd": []})
    m["rd"].append(x)
for m in MAT.values():
    m["n"] = len(m["rd"])
    m["complete"] = m["n"] == m["tA"] + m["tB"] and m["n"] <= 24 and [x["round"] for x in m["rd"]] == list(range(1, m["n"] + 1))


def step(x, y):
    if None in (x["tb"], x["cb"], y["tb"], y["cb"]):
        return "missing"
    tT = 1 if x["winner"] == T else 0
    tC = 1 - tT
    ns = (y["tb"] == x["tb"] + tT) and (y["cb"] == x["cb"] + tC)
    sw = (y["tb"] == x["cb"] + tC) and (y["cb"] == x["tb"] + tT)
    return "both" if (ns and sw) else "noswap" if ns else "swap" if sw else "ANOMALY"


def d8_sig(m):
    live = {x["round"]: (x["tb"], x["cb"]) for x in m["rd"] if x["tb"] is not None}
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


def show(key, m):
    rd = m["rd"]
    print(f"\n{key.split('/')[3]}/{key.split('/')[4][:8]} map={m['map']} n={m['n']} tA={m['tA']} tB={m['tB']} "
          f"d8_sig={d8_sig(m)}")
    print(f"  {'r':>3}{'winner':>7}{'tb_fe':>7}{'cb_fe':>7}  step      bside")
    for i, x in enumerate(rd):
        st = step(x, rd[i + 1]) if i + 1 < len(rd) and rd[i + 1]["round"] == x["round"] + 1 else "-"
        w = "T" if x["winner"] == T else "CT"
        bs = "" if x["bside"] is None else ("CT" if x["bside"] == CT else "T")
        print(f"  {x['round']:>3}{w:>7}{str(x['tb']):>7}{str(x['cb']):>7}  {st:<9} {bs}")


# (a) () complete matches with step 13 = swap
grpA = []
for k, m in MAT.items():
    if m["complete"] and d8_sig(m) == ():
        st13 = [step(x, y) for x, y in zip(m["rd"][:-1], m["rd"][1:]) if x["round"] == 13]
        if st13 and st13[0] == "swap":
            grpA.append((k, m))
print(f"(a) () complete matches with step 13 = swap: {len(grpA)}")
for k, m in grpA[:2]:
    show(k, m)

# (b) () complete matches that are other_misfit under MR12 (r13 flip does not fit)
grpB = []
for k, m in MAT.items():
    if not (m["complete"] and d8_sig(m) == ()):
        continue
    rd = m["rd"]
    wa = sum(1 for x in rd if x["winner"] == (T if x["round"] <= 12 else CT))
    if (wa, m["n"] - wa) == (m["tA"], m["tB"]):
        continue
    r13 = next((x for x in rd if x["round"] == 13), None)
    if r13 is not None:
        nwa = wa - 1 if r13["winner"] == CT else wa + 1
        if (nwa, m["n"] - nwa) == (m["tA"], m["tB"]):
            continue
    grpB.append((k, m))
print(f"\n(b) () complete other_misfit (MR12 and r13 flip both fail): {len(grpB)}")
for k, m in grpB[:2]:
    show(k, m)

# (c) (8,) complete other_misfit that a round-9 flip does not reconcile
grpC = []
for k, m in MAT.items():
    if not (m["complete"] and d8_sig(m) == (8,)):
        continue
    rd = m["rd"]
    fit_h = lambda flip: (sum((1 if ((x["winner"] == (T if x["round"] <= 8 else CT)) != (x["round"] == flip))
                               else 0) for x in rd), m["n"])
    wa = sum(1 for x in rd if x["winner"] == (T if x["round"] <= 8 else CT))
    if (wa, m["n"] - wa) == (m["tA"], m["tB"]):
        continue
    wa9 = sum(1 for x in rd if x["winner"] == (T if x["round"] <= 8 else CT)) - (
        1 if next(x for x in rd if x["round"] == 9)["winner"] == (T if 9 <= 8 else CT) else -1)
    if (wa9, m["n"] - wa9) == (m["tA"], m["tB"]):
        continue
    grpC.append((k, m))
print(f"\n(c) (8,) complete matches not reconciled by a round-9 flip: {len(grpC)}")
for k, m in grpC[:2]:
    show(k, m)

print("\nPROBE2 DONE")
