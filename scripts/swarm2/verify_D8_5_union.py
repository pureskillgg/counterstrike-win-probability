"""verify_D8_5_union: placebo baseline for the 'flip round h+1, normal or swapped orientation' fit (D8-5 verifier).

For complete matches announced MR8 (h=8) and MR12 (h=12), for every candidate flip round r, the share of matches that
reconcile exactly with (a) normal orientation, (b) swapped orientation, (c) either. Round h+1 should stand out if the
first post-halftime round is the defect. Appends to logs\\swarm2\\verify_D8_5.log.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D8_5_union.py
"""
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
    m = MAT.setdefault(x["match_key"], {"tA": int(x["tA"]), "tB": int(x["tB"]), "rd": []})
    m["rd"].append(x)


def complete(m):
    n = len(m["rd"])
    return n == m["tA"] + m["tB"] and n <= 24 and [x["round"] for x in m["rd"]] == list(range(1, n + 1))


def fit(m, h, flip, orient):
    wa = 0
    for x in m["rd"]:
        won = x["winner"] == (T if x["round"] <= h else CT)
        if flip is not None and x["round"] == flip:
            won = not won
        wa += int(won)
    n = len(m["rd"])
    tA, tB = (m["tB"], m["tA"]) if orient else (m["tA"], m["tB"])
    return (wa, n - wa) == (tA, tB)


say("\n=== verify_D8_5 UNION PLACEBO (appended): flip r, normal or swapped orientation, complete matches ===")
for label, a, h, rounds in [("MR8 (h=8)", [7], 8, range(1, 17)), ("MR12 (h=12)", [11], 12, range(1, 25))]:
    grp = [m for k, m in MAT.items() if complete(m) and ANN.get(k) == a]
    n = len(grp)
    say(f"\n{label}: complete matches n={n}")
    say(f"  {'flip r':>7}{'normal':>9}{'swapped':>10}{'either':>9}")
    rows = {}
    for r in rounds:
        cn = sum(1 for m in grp if fit(m, h, r, False))
        cs = sum(1 for m in grp if fit(m, h, r, True))
        ce = sum(1 for m in grp if fit(m, h, r, False) or fit(m, h, r, True))
        rows[r] = (cn, cs, ce)
        say(f"  {r:>7}{cn:>9}{cs:>10}{ce:>9}" + ("   <-- h+1" if r == h + 1 else ""))
    others = [rows[r][2] for r in rounds if r != h + 1]
    say(f"  placebo 'either' over other rounds: min {min(others)}, median {sorted(others)[len(others) // 2]}, "
        f"max {max(others)}; round h+1 'either' = {rows[h + 1][2]} of {n}")
    # matches with tA == tB or |tA - tB| <= 1 cannot tell the two orientations apart
    close = sum(1 for m in grp if abs(m["tA"] - m["tB"]) <= 1)
    say(f"  matches with |tA-tB| <= 1 (orientation barely identifiable): {close} of {n}")
say("UNION DONE")
LOGF.close()
print("UNION DONE")
