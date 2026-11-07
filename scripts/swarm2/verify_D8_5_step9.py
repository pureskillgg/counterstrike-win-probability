"""verify_D8_5_step9: scoreboard step census keyed on the announced halftime (D8-5 verifier).

For matches announced as MR8 (halftime after round 8) and MR12 (after round 12), the winner-consistent step census
(a freeze-end-to-freeze-end step must match no-swap or swap given the round_end winner) at the steps after round
h-1, h and h+1, including the 52 MR8 matches that the D8 scoreboard detector missed. Appends to the log.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D8_5_step9.py
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
OUT = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
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


def steps(m):
    rd = m["rd"]
    out = {}
    for x, y in zip(rd[:-1], rd[1:]):
        if y["round"] != x["round"] + 1:
            out[x["round"]] = "gap"
            continue
        if None in (x["tb"], x["cb"], y["tb"], y["cb"]):
            out[x["round"]] = "missing"
            continue
        tT = 1 if x["winner"] == T else 0
        tC = 1 - tT
        ns = (y["tb"] == x["tb"] + tT) and (y["cb"] == x["cb"] + tC)
        sw = (y["tb"] == x["cb"] + tC) and (y["cb"] == x["tb"] + tT)
        out[x["round"]] = "both" if (ns and sw) else "noswap" if ns else "swap" if sw else "anomaly"
    return out


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


say("\n=== verify_D8_5 STEP CENSUS BY ANNOUNCED HALFTIME (appended) ===")
say("steps are labelled by the round whose winner they test; h = last round of first half (MR8 h=8, MR12 h=12)")
for label, a, h in [("MR8-announced (ann [7], h=8)", [7], 8), ("MR12-announced (ann [11], h=12)", [11], 12)]:
    grp = [(k, m) for k, m in MAT.items() if m["day"] >= 12 and ANN.get(k) == a]
    say(f"\n{label}: all matches days 12-25 = {len(grp)}")
    for step_r in [h - 1, h, h + 1, h + 2]:
        c = collections.Counter()
        for k, m in grp:
            st = steps(m).get(step_r)
            c[st if st is not None else "absent"] += 1
        tot = sum(v for kk, v in c.items() if kk not in ("absent",))
        rate = c["anomaly"] / tot if tot else float("nan")
        say(f"  step after round {step_r:>2} (h{step_r - h:+d}): " +
            "  ".join(f"{kk}={v}" for kk, v in sorted(c.items())) + f"   anomaly rate={rate:.3f}")
    # D8-missed subset
    missed = [(k, m) for k, m in grp if d8_sig(m) == ()]
    c = collections.Counter(steps(m).get(h + 1, "absent") for k, m in missed)
    say(f"  of which D8 sig () (scoreboard detector missed the halftime): n={len(missed)}; step after h+1 "
        + "  ".join(f"{kk}={v}" for kk, v in sorted(c.items())))
    det = [(k, m) for k, m in grp if d8_sig(m) == (h,)] if h == 8 else []
    if det:
        c2 = collections.Counter(steps(m).get(h + 1, "absent") for k, m in det)
        say(f"  of which D8 sig (8,): n={len(det)}; step after h+1 " + "  ".join(f"{kk}={v}" for kk, v in sorted(c2.items())))
say("\nSTEP9 DONE")
LOGF.close()
print("STEP9 DONE")
