"""verify_D8_5_xtab: cross-tab of the scoreboard step at the first post-halftime round against header fit (D8-5).

For complete matches announced as MR8 (h=8) and MR12 (h=12): the step category after round h+1 (anomaly / noswap /
swap / both, from scoreboard vs round_end winner) against whether the header reconciles with no flip (F0) or with
a flip of round h+1 (F1), normal orientation. If the scoreboard identifies which source is inverted, the cross-tab
shows F1 concentrated in anomaly matches and F0 in noswap matches. Appends to the log.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D8_5_xtab.py
"""
import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
LOG = ROOT / "logs" / "swarm2" / "verify_D8_5.log"
OUT = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D8_5")
T, CT = 2, 3
LOGF = open(LOG, "a", encoding="utf-8", newline="\n")


def say(*args):
    s = " ".join(str(a) for a in args)
    print(s)
    LOGF.write(s + "\n")
    LOGF.flush()


RS = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "event_type"])
                for f in sorted(glob.glob(str(BASE / f"basic_round_state{SUF}" / "dataframe_*")))],
               how="vertical_relaxed")
ann = (RS.filter(pl.col("event_type") == "round_announce_last_round_half")
       .select(["match_key", "round"]).unique().group_by("match_key").agg(pl.col("round").sort()))
ANN = {r["match_key"]: r["round"] for r in ann.iter_rows(named=True)}
R = pl.read_parquet(OUT / "rounds.parquet").sort(["match_key", "round"])
MAT = {}
for x in R.iter_rows(named=True):
    m = MAT.setdefault(x["match_key"], {"map": x["map_name"], "tA": int(x["tA"]), "tB": int(x["tB"]),
                                        "day": int(x["day"]), "rd": []})
    m["rd"].append(x)


def complete(m):
    n = len(m["rd"])
    return n == m["tA"] + m["tB"] and n <= 24 and [x["round"] for x in m["rd"]] == list(range(1, n + 1))


def step_cat(m, r):
    rd = m["rd"]
    for x, y in zip(rd[:-1], rd[1:]):
        if x["round"] != r:
            continue
        if y["round"] != r + 1 or None in (x["tb"], x["cb"], y["tb"], y["cb"]):
            return "missing"
        tT = 1 if x["winner"] == T else 0
        tC = 1 - tT
        ns = (y["tb"] == x["tb"] + tT) and (y["cb"] == x["cb"] + tC)
        sw = (y["tb"] == x["cb"] + tC) and (y["cb"] == x["tb"] + tT)
        return "both" if (ns and sw) else "noswap" if ns else "swap" if sw else "anomaly"
    return "absent"


def fit(m, h, flip, orient=False):
    wa = 0
    for x in m["rd"]:
        won = x["winner"] == (T if x["round"] <= h else CT)
        if x["round"] == flip:
            won = not won
        wa += int(won)
    n = len(m["rd"])
    tA, tB = (m["tB"], m["tA"]) if orient else (m["tA"], m["tB"])
    return (wa, n - wa) == (tA, tB)


say("\n=== verify_D8_5 CROSS-TAB at first post-halftime round (appended) ===")
for label, a, h in [("MR8 (h=8, first post-halftime round 9)", [7], 8),
                    ("MR12 (h=12, first post-halftime round 13)", [11], 12)]:
    grp = [m for k, m in MAT.items() if complete(m) and ANN.get(k) == a]
    tab = collections.Counter()
    for m in grp:
        cat = step_cat(m, h + 1)
        tab[(cat, "F0" if fit(m, h, None) else "-", "F1" if fit(m, h, h + 1) else "-",
             "O1" if fit(m, h, h + 1, orient=True) else "-")] += 1
    say(f"\n{label}: complete matches n={len(grp)}")
    say(f"  {'step cat':<9}{'F0 fits':>9}{'F1 fits':>9}{'O1 fits':>9}{'F1 or O1':>10}{'total':>8}")
    for cat in ["anomaly", "noswap", "swap", "both", "missing", "absent"]:
        f0 = sum(v for (c, f, g, o), v in tab.items() if c == cat and f == "F0")
        f1 = sum(v for (c, f, g, o), v in tab.items() if c == cat and g == "F1")
        o1 = sum(v for (c, f, g, o), v in tab.items() if c == cat and o == "O1")
        fo = sum(v for (c, f, g, o), v in tab.items() if c == cat and (g == "F1" or o == "O1"))
        tot = sum(v for (c, f, g, o), v in tab.items() if c == cat)
        if tot:
            say(f"  {cat:<9}{f0:>9}{f1:>9}{o1:>9}{fo:>10}{tot:>8}")
say("\nFORMAT NOTE: F0 = no flip (header vs round_end as recorded, swap after h); F1 = flip round h+1.")
say("XTAB DONE")
LOGF.close()
print("XTAB DONE")
