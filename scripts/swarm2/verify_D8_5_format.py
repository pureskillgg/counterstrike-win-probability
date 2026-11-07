"""verify_D8_5_format: re-test the header fit keyed on the game's announced halftime (D8-5 verifier).

Halftime from the round_state announcement round_announce_last_round_half: value v means the last round of the
first half is v+1 (MR12: v=11 -> h=12; MR8: v=7 -> h=8). Compared with the D8 scoreboard signature.
For each complete match with exactly one first-half announcement, tests exact header fit under:
  F0  swap after h, round_end winners as recorded, header orientation as recorded (tA = team starting T)
  F1  swap after h, flip round h+1 winner (first post-halftime round)
  O0  as F0 but header orientation swapped (tA <-> tB)
  O1  as F1 but header orientation swapped
  M12 MR12 swap at 12 with round 13 flip (the claim's D4 rule), for comparison
Appends to logs\\swarm2\\verify_D8_5.log.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D8_5_format.py
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


RS = read_dir(BASE / f"basic_round_state{SUF}", ["match_key", "round", "event_type"])
ann = (RS.filter(pl.col("event_type") == "round_announce_last_round_half")
       .select(["match_key", "round"]).unique().group_by("match_key").agg(pl.col("round").sort()))
ANN = {r["match_key"]: r["round"] for r in ann.iter_rows(named=True)}

R = pl.read_parquet(OUT / "rounds.parquet").sort(["match_key", "round"])
MAT = {}
for x in R.iter_rows(named=True):
    m = MAT.setdefault(x["match_key"], {"map": x["map_name"], "tA": int(x["tA"]), "tB": int(x["tB"]),
                                        "day": int(x["day"]), "rd": []})
    m["rd"].append(x)


def fit(m, h, flip=None, orient=False, n_swap=True):
    rd = m["rd"]
    wa = 0
    for x in rd:
        side_a = T if (h is None or x["round"] <= h) else CT
        won = x["winner"] == side_a
        if flip is not None and x["round"] == flip:
            won = not won
        wa += int(won)
    n = len(rd)
    tA, tB = (m["tB"], m["tA"]) if orient else (m["tA"], m["tB"])
    return (wa, n - wa) == (tA, tB)


def complete(m):
    n = len(m["rd"])
    return n == m["tA"] + m["tB"] and n <= 24 and [x["round"] for x in m["rd"]] == list(range(1, n + 1))


def cls_d4(m):
    rd = m["rd"]
    n = len(rd)
    mm13 = any(x["round"] == 13 and x["bside"] is not None and x["bside"] != x["winner"] for x in rd)
    if mm13:
        return "affected"
    wa = sum(1 for x in rd if x["winner"] == (T if x["round"] <= 12 else CT))
    if (wa, n - wa) == (m["tA"], m["tB"]):
        return "fits"
    r13 = next((x for x in rd if x["round"] == 13), None)
    if r13 is not None:
        nwa = wa - 1 if r13["winner"] == CT else wa + 1
        if (nwa, n - nwa) == (m["tA"], m["tB"]):
            return "r13_flip_misfit"
    return "other_misfit"


say("\n=== verify_D8_5 FORMAT CHECK (appended): halftime from round_announce_last_round_half ===")
say("ASSUMPTION: announcement value v => last round of first half is v+1 (verified: MR12 announcements at 11,"
    " MR8 at 7; complete (8,) matches all carry 7)")
rows = []
for k, m in MAT.items():
    if not complete(m):
        continue
    a = ANN.get(k, [])
    if len(a) != 1 or a[0] not in (7, 11):
        cat = f"ann={a}"
        h = None
    else:
        h = a[0] + 1
        cat = f"h={h}"
    m["h"] = h
    rows.append({"key": k, "map": m["map"], "day": m["day"], "cat": cat, "h": h if h is not None else -1,
                 "cls": cls_d4(m),
                 "F0": fit(m, h) if h else None, "F1": fit(m, h, flip=h + 1) if h else None,
                 "O0": fit(m, h, orient=True) if h else None,
                 "O1": fit(m, h, flip=h + 1, orient=True) if h else None,
                 "M12": fit(m, 12, flip=13) if any(x["round"] == 13 for x in m["rd"]) else None})
Q = pl.DataFrame(rows)
say(f"complete matches: {Q.height}")
say("announcement categories among complete matches:")
say(Q.group_by("cat").len().sort("len", descending=True))
say("\nclass x halftime (announced), complete matches; counts of exact fit under F0 / F1 / O0 / O1 / M12:")
say(Q.group_by(["cls", "h"]).agg(pl.len().alias("n"), pl.col("F0").sum().alias("F0"), pl.col("F1").sum().alias("F1"),
                                  pl.col("O0").sum().alias("O0"), pl.col("O1").sum().alias("O1"),
                                  pl.col("M12").sum().alias("M12")).sort(["cls", "h"]))
say("\nother_misfit by announced halftime and map:")
om = Q.filter(pl.col("cls") == "other_misfit")
say(om.group_by(["h", "map"]).agg(pl.len().alias("n"), pl.col("F1").sum().alias("F1_fit"),
                                   pl.col("O0").sum().alias("O0_fit"), pl.col("O1").sum().alias("O1_fit")).sort(["h", "map"]))
say("\nfits by class for the MR8 group (h=8) and MR12 group (h=12): F1 and O1 coverage")
say(Q.filter(pl.col("h") == 8).group_by("cls").agg(pl.len().alias("n"), pl.col("F0").sum().alias("F0"),
                                                    pl.col("F1").sum().alias("F1"), pl.col("O0").sum().alias("O0"),
                                                    pl.col("O1").sum().alias("O1")).sort("cls"))
say(Q.filter(pl.col("h") == 12).group_by("cls").agg(pl.len().alias("n"), pl.col("F0").sum().alias("F0"),
                                                     pl.col("F1").sum().alias("F1"), pl.col("O0").sum().alias("O0"),
                                                     pl.col("O1").sum().alias("O1")).sort("cls"))
any_fit = Q.with_columns(
    pl.when(pl.col("F0") | pl.col("F1") | pl.col("O0") | pl.col("O1")).then(pl.lit(True)).otherwise(pl.lit(False))
    .alias("any_fit"))
say("\nmatches (announced-halftime set) with no exact fit under F0/F1/O0/O1 (cls, h, n):")
say(any_fit.filter(~pl.col("any_fit")).group_by(["cls", "h"]).len().sort(["cls", "h"]))
say("\nFORMAT DONE")
LOGF.close()
print("FORMAT DONE")
