"""verify_D8_5_build_check: build-number confound check (D8-5 verifier, independent of other blocks in the log).

Reads header build_num (read-only), joins it to the scratchpad per-match table, and tabulates for days 12-25 and
1-7: header matches, complete matches, announced MR8/MR12, other_misfit (MR12 rule), first post-halftime scoreboard
anomaly rate, and round-13 bomb mismatch for planted rows. Appends to logs\\swarm2\\verify_D8_5.log.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D8_5_build_check.py
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


HS = pl.read_parquet_schema(sorted(glob.glob(str(BASE / "header" / "dataframe_*")))[0])
say("\n=== verify_D8_5 BUILD CONFOUND CHECK (appended, verify_D8_5_build_check.py) ===")
say("header columns:", list(HS.keys()))
if "build_num" not in HS:
    say("build_num not in header; check aborted")
    LOGF.close()
    raise SystemExit(0)
H = read_dir(BASE / "header", ["key", "build_num"])
BLD = {r["key"]: r["build_num"] for r in H.iter_rows(named=True)}
say("header build_num counts (all header matches):", H.group_by("build_num").len().sort("build_num").rows())

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
        if None in (x["tb"], x["cb"], y["tb"], y["cb"]):
            return "missing"
        tT = 1 if x["winner"] == T else 0
        tC = 1 - tT
        ns = (y["tb"] == x["tb"] + tT) and (y["cb"] == x["cb"] + tC)
        sw = (y["tb"] == x["cb"] + tC) and (y["cb"] == x["tb"] + tT)
        return "both" if (ns and sw) else "noswap" if ns else "swap" if sw else "anomaly"
    return "absent"


def d4_other_misfit(m):
    rd, n, tA, tB = m["rd"], len(m["rd"]), m["tA"], m["tB"]
    mm13 = any(x["round"] == 13 and x["bside"] is not None and x["bside"] != x["winner"] for x in rd)
    wa = sum(1 for x in rd if x["winner"] == (T if x["round"] <= 12 else CT))
    if (wa, n - wa) == (tA, tB) or mm13:
        return False
    r13 = next((x for x in rd if x["round"] == 13), None)
    if r13 is not None:
        nwa = wa - 1 if r13["winner"] == CT else wa + 1
        if (nwa, n - nwa) == (tA, tB):
            return False
    return True


for band, lo, hi in [("days 12-25", 12, 25), ("days 01-07", 1, 7)]:
    say(f"\n{band}:")
    tab = collections.defaultdict(collections.Counter)
    for k, m in MAT.items():
        if not (lo <= m["day"] <= hi):
            continue
        b = BLD.get(k)
        c = tab[b]
        c["header_matches"] += 1
        c["complete"] += int(complete(m))
        a = ANN.get(k, [])
        c["MR8_ann"] += int(a == [7])
        c["MR12_ann"] += int(a == [11])
        if complete(m):
            c["complete_MR8"] += int(a == [7])
            c["complete_MR12"] += int(a == [11])
            c["other_misfit"] += int(d4_other_misfit(m))
    say(f"  {'build':>7}{'hdr':>7}{'complete':>10}{'cmpl MR8':>10}{'cmpl MR12':>11}{'other_mis':>11}{'MR8 ann':>9}{'MR12 ann':>10}")
    for b in sorted(tab, key=lambda z: str(z)):
        c = tab[b]
        say(f"  {str(b):>7}{c['header_matches']:>7}{c['complete']:>10}{c['complete_MR8']:>10}{c['complete_MR12']:>11}"
            f"{c['other_misfit']:>11}{c['MR8_ann']:>9}{c['MR12_ann']:>10}")

say("\nfirst post-halftime scoreboard step, announced formats, days 12-25 (anomaly / all with a step):")
for b in sorted({BLD.get(k) for k in MAT}, key=lambda z: str(z)):
    for label, a, h in [("MR8 (step 9)", [7], 8), ("MR12 (step 13)", [11], 12)]:
        cats = collections.Counter()
        for k, m in MAT.items():
            if m["day"] < 12 or BLD.get(k) != b or ANN.get(k) != a:
                continue
            cats[step_cat(m, h + 1)] += 1
        tot = sum(v for kk, v in cats.items() if kk not in ("absent", "missing"))
        if tot:
            say(f"  build {str(b):>5} {label:<15} anomaly={cats['anomaly']:>4} of {tot:>4} "
                f"({cats['anomaly'] / tot:.3f})  other: " +
                ", ".join(f"{kk}={v}" for kk, v in sorted(cats.items()) if kk not in ("anomaly",)))

say("\nround-13 bomb side vs round_end winner, planted rows, days 12-25, by build:")
BS = read_dir(BASE / f"basic_bomb_state{SUF}", ["match_key", "round", "event_type"]).filter(
    pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
bo = BS.group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")).with_columns(
    pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(CT)).otherwise(pl.lit(T)).alias("bside"))
bod = {(r["match_key"], r["round"]): r["bside"] for r in bo.iter_rows(named=True)}
bt = collections.defaultdict(lambda: [0, 0])
for k, m in MAT.items():
    if m["day"] < 12:
        continue
    for x in m["rd"]:
        if x["round"] == 13 and (k, 13) in bod:
            bt[BLD.get(k)][0] += 1
            bt[BLD.get(k)][1] += int(bod[(k, 13)] != x["winner"])
for b, (n_, mm_) in sorted(bt.items(), key=lambda z: str(z[0])):
    say(f"  build {str(b):>5}: round-13 planted rows {n_}, mismatches {mm_} ({mm_ / n_ if n_ else float('nan'):.3f})")
say("BUILD CHECK DONE")
LOGF.close()
print("BUILD CHECK DONE")
