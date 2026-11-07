"""verify_D8_5_build_steps: are round labels aligned with the scoreboard in build 9842? (D8-5 verifier)

For announced MR12 and MR8 matches in days 12-25, by build, the step census at the halftime step (label h) and at the
steps before and after it. If build 9842 labels were offset from the scoreboard, the halftime step would not show a
swap there. Appends to logs\\swarm2\\verify_D8_5.log.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D8_5_build_steps.py
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


def read_dir(fd, cols):
    fs = sorted(glob.glob(str(fd / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


H = read_dir(BASE / "header", ["key", "build_num"])
BLD = {r["key"]: r["build_num"] for r in H.iter_rows(named=True)}
RS = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "event_type"])
                for f in sorted(glob.glob(str(BASE / f"basic_round_state{SUF}" / "dataframe_*")))],
               how="vertical_relaxed")
ann = (RS.filter(pl.col("event_type") == "round_announce_last_round_half")
       .select(["match_key", "round"]).unique().group_by("match_key").agg(pl.col("round").sort()))
ANN = {r["match_key"]: r["round"] for r in ann.iter_rows(named=True)}
R = pl.read_parquet(OUT / "rounds.parquet").sort(["match_key", "round"])
MAT = {}
for x in R.iter_rows(named=True):
    m = MAT.setdefault(x["match_key"], {"rd": []})
    m["rd"].append(x)


def step_cat(m, r):
    rd = m["rd"]
    for x, y in zip(rd[:-1], rd[1:]):
        if x["round"] != r:
            continue
        if None in (x["tb"], x["cb"], y["tb"], y["cb"]):
            return "missing"
        tT = 1 if x["winner"] == T else 0
        tC = 1 - tT
        ns = (y["tb"] == x["tb"] + tT) and (y["cb"] == x["cb"] + tC)
        sw = (y["tb"] == x["cb"] + tC) and (y["cb"] == x["tb"] + tT)
        return "both" if (ns and sw) else "noswap" if ns else "swap" if sw else "anomaly"
    return "absent"


say("\n=== verify_D8_5 BUILD STEP ALIGNMENT (appended, verify_D8_5_build_steps.py) ===")
say("steps labelled r test round r's winner between freeze_end r and r+1; halftime step is r = h")
for label, a, h in [("MR8 (h=8)", [7], 8), ("MR12 (h=12)", [11], 12)]:
    for b in sorted({BLD.get(k) for k in MAT}, key=lambda z: str(z)):
        keys = [k for k in MAT if BLD.get(k) == b and ANN.get(k) == a and int(k.split("/")[3]) >= 12]
        if not keys:
            continue
        out = []
        for r in [h - 1, h, h + 1]:
            c = collections.Counter(step_cat(MAT[k], r) for k in keys)
            out.append(f"step {r} (h{r - h:+d}): " + " ".join(f"{kk}={v}" for kk, v in sorted(c.items())))
        say(f"  {label} build {b} n={len(keys)}:  " + " | ".join(out))
say("BUILD STEPS DONE")
LOGF.close()
print("BUILD STEPS DONE")
