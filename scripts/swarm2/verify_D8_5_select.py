"""verify_D8_5_select: where do the clean-step MR12 matches go? (D8-5 verifier, selection check)

Among MR12-announced matches (days 12-25), the step after round 13 is either clean (no-swap consistent) or not.
The complete-regulation universe contains no clean-step MR12 match. This script shows the round-count shortfall
(n - (tA+tB)) for clean versus non-clean matches, so the complete-only rates can be read correctly.
Appends to logs\\swarm2\\verify_D8_5.log.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D8_5_select.py
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
    m = MAT.setdefault(x["match_key"], {"tA": int(x["tA"]), "tB": int(x["tB"]), "day": int(x["day"]), "rd": []})
    m["rd"].append(x)


def step13(m):
    rd = m["rd"]
    for x, y in zip(rd[:-1], rd[1:]):
        if x["round"] != 13:
            continue
        if None in (x["tb"], x["cb"], y["tb"], y["cb"]):
            return "missing"
        tT = 1 if x["winner"] == T else 0
        tC = 1 - tT
        ns = (y["tb"] == x["tb"] + tT) and (y["cb"] == x["cb"] + tC)
        sw = (y["tb"] == x["cb"] + tC) and (y["cb"] == x["tb"] + tT)
        return "both" if (ns and sw) else "noswap" if ns else "swap" if sw else "anomaly"
    return "absent"


say("\n=== verify_D8_5 SELECTION CHECK (appended): MR12-announced, days 12-25 ===")
grp = [(k, m) for k, m in MAT.items() if m["day"] >= 12 and ANN.get(k) == [11]]
tab = collections.defaultdict(collections.Counter)
for k, m in grp:
    cat = step13(m)
    shortfall = len(m["rd"]) - (m["tA"] + m["tB"])
    tab[cat][shortfall] += 1
for cat in ["noswap", "swap", "both", "anomaly", "absent", "missing"]:
    if cat in tab:
        tot = sum(tab[cat].values())
        say(f"  step13={cat:<8} matches={tot:>5}  rounds minus header total (value: count): "
            f"{sorted(tab[cat].items(), key=lambda z: -z[1])[:6]}")
say("  (complete = shortfall 0 and n<=24 and contiguous; clean-step matches have shortfall != 0 by construction of the complete set)")
LOGF.close()
print("SELECT DONE")
