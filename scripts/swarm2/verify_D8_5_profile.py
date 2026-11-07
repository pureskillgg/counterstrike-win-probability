"""verify_D8_5_profile: single-round-flip profile against a chance baseline (D8-5 verifier).

For complete matches grouped by announced halftime (MR8: h=8, MR12: h=12), for every candidate round r and each
header orientation, the share of matches whose header totals reconcile exactly after flipping round r's winner
(under swap after h). A real defect at the first post-halftime round should show a clear peak at r = h+1;
other rounds give the chance baseline. Appends to logs\\swarm2\\verify_D8_5.log.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D8_5_profile.py
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


def complete(m):
    n = len(m["rd"])
    return n == m["tA"] + m["tB"] and n <= 24 and [x["round"] for x in m["rd"]] == list(range(1, n + 1))


def fit_flip(m, h, flip, orient):
    wa = 0
    for x in m["rd"]:
        side_a = T if x["round"] <= h else CT
        won = x["winner"] == side_a
        if x["round"] == flip:
            won = not won
        wa += int(won)
    n = len(m["rd"])
    tA, tB = (m["tB"], m["tA"]) if orient else (m["tA"], m["tB"])
    return (wa, n - wa) == (tA, tB)


def fit_noflip(m, h, orient):
    return fit_flip(m, h, None, orient)


def profile(label, group, h, rounds):
    n = len(group)
    say(f"\n{label}: n={n} (halftime h={h}, swap after round h)")
    say(f"  {'flip round':>10}{'normal ori.':>14}{'swapped ori.':>15}")
    best = {}
    for r in rounds:
        cn = sum(1 for m in group if fit_flip(m, h, r, False))
        cs = sum(1 for m in group if fit_flip(m, h, r, True))
        best[r] = (cn, cs)
        mark = "  <-- h+1" if r == h + 1 else ""
        say(f"  {r:>10}{cn:>10} ({cn / n:5.2f}){cs:>9} ({cs / n:5.2f}){mark}")
    cn0 = sum(1 for m in group if fit_noflip(m, h, False))
    cs0 = sum(1 for m in group if fit_noflip(m, h, True))
    say(f"  no flip: normal {cn0} ({cn0 / n:.2f}), swapped {cs0} ({cs0 / n:.2f})")
    return best


say("\n=== verify_D8_5 SINGLE-FLIP PROFILE (appended) ===")
mr8 = [m for k, m in MAT.items() if complete(m) and ANN.get(k) == [7]]
mr12 = [m for k, m in MAT.items() if complete(m) and ANN.get(k) == [11]]
say(f"complete MR8-announced (ann=[7]): {len(mr8)}; complete MR12-announced (ann=[11]): {len(mr12)}")
say("ASSUMPTION: announcement [v] => first half ends after round v+1; profile uses that halftime.")
b8 = profile("MR8-announced complete matches", mr8, 8, range(1, 17))
b12 = profile("MR12-announced complete matches (control: correct halftime 12)", mr12, 12, range(1, 25))

# chance baseline for MR8: median fit count over placebo rounds (not h+1), per orientation
pl_n = [b8[r][0] for r in b8 if r != 9]
pl_s = [b8[r][1] for r in b8 if r != 9]
say(f"\nMR8 placebo rounds (not 9): normal orientation counts min/median/max = "
    f"{min(pl_n)}/{sorted(pl_n)[len(pl_n) // 2]}/{max(pl_n)}; swapped = "
    f"{min(pl_s)}/{sorted(pl_s)[len(pl_s) // 2]}/{max(pl_s)}")
say(f"MR8 round 9: normal {b8[9][0]}, swapped {b8[9][1]}")
pl12n = [b12[r][0] for r in b12 if r != 13]
pl12s = [b12[r][1] for r in b12 if r != 13]
say(f"MR12 placebo rounds (not 13): normal min/median/max = {min(pl12n)}/{sorted(pl12n)[len(pl12n) // 2]}/"
    f"{max(pl12n)}; swapped = {min(pl12s)}/{sorted(pl12s)[len(pl12s) // 2]}/{max(pl12s)}")
say(f"MR12 round 13: normal {b12[13][0]}, swapped {b12[13][1]}")
say("\nPROFILE DONE")
LOGF.close()
print("PROFILE DONE")
