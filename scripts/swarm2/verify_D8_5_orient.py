"""verify_D8_5_orient: header orientation from the scoreboard's final side scores (D8-5 verifier).

After one halftime swap the T side at the end of the match is the team that started on CT. The final side scores are
the freeze-end scores of the last round plus that round's round_end winner (no swap follows the last round). Compare
(final T-side score, final CT-side score) with the header totals:
  normal orientation (header tA = team that started T)  -> (final T, final CT) = (tB, tA)
  swapped orientation                                   -> (final T, final CT) = (tA, tB)
This uses the last round only, so it does not depend on the round-9 (or round-13) winner that the flip model
changes. Run for MR8 (one swap after 8) and MR12 (one swap after 12). Appends to the log.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D8_5_orient.py
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
    m = MAT.setdefault(x["match_key"], {"tA": int(x["tA"]), "tB": int(x["tB"]), "rd": []})
    m["rd"].append(x)


def complete(m):
    n = len(m["rd"])
    return n == m["tA"] + m["tB"] and n <= 24 and [x["round"] for x in m["rd"]] == list(range(1, n + 1))


say("\n=== verify_D8_5 ORIENTATION FROM FINAL SIDE SCORES (appended) ===")
for label, a in [("MR8 (one swap after 8)", [7]), ("MR12 (one swap after 12)", [11])]:
    grp = [m for k, m in MAT.items() if complete(m) and ANN.get(k) == a]
    c = collections.Counter()
    for m in grp:
        last = m["rd"][-1]
        if last["tb"] is None or last["cb"] is None:
            c["no final score"] += 1
            continue
        sT = last["tb"] + (1 if last["winner"] == T else 0)
        sC = last["cb"] + (1 if last["winner"] == CT else 0)
        if sT + sC != len(m["rd"]):
            c["sum != n"] += 1
            continue
        if (sT, sC) == (m["tB"], m["tA"]) and (sT, sC) == (m["tA"], m["tB"]):
            c["both (tA==tB)"] += 1
        elif (sT, sC) == (m["tB"], m["tA"]):
            c["normal"] += 1
        elif (sT, sC) == (m["tA"], m["tB"]):
            c["swapped"] += 1
        else:
            c["neither"] += 1
    say(f"{label}: complete n={len(grp)}  " + "  ".join(f"{k}={v}" for k, v in sorted(c.items())))
say("ORIENT DONE")
LOGF.close()
print("ORIENT DONE")
