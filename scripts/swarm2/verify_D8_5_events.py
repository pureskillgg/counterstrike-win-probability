"""verify_D8_5_events: independent halftime check from the game's own round-announcement events (D8-5 verifier).

If the (8,) regime is a real halftime after round 8, the game should announce the last round of the half
(round_announce_last_round_half) at round 8 in those matches; for (12,) matches at round 12. Compared with the
D8 signature and with the header-fit classes. Also lists the round_state column schema for the record.
Appends to logs\\swarm2\\verify_D8_5.log.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D8_5_events.py
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


def read_dir(fd, cols=None):
    fs = sorted(glob.glob(str(fd / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


say("\n=== verify_D8_5 EVENT CHECK (appended) ===")
RS_schema = pl.read_parquet_schema(sorted(glob.glob(str(BASE / f"basic_round_state{SUF}" / "dataframe_*")))[0])
say("round_state columns:", list(RS_schema.keys()))
RE_schema = pl.read_parquet_schema(sorted(glob.glob(str(BASE / f"basic_round_end{SUF}" / "dataframe_*")))[0])
say("round_end columns:", list(RE_schema.keys()))

RS = read_dir(BASE / f"basic_round_state{SUF}", ["match_key", "round", "tick", "event_type"])
EV = ["round_announce_last_round_half", "round_announce_match_point", "round_announce_final",
      "round_announce_match_start", "cs_win_panel_match"]
ev = (RS.filter(pl.col("event_type").is_in(EV)).select(["match_key", "round", "event_type"]).unique())
say("announcement event rows by type:", ev.group_by("event_type").len().sort("event_type").rows())

rounds_by_m = {}
for r in ev.iter_rows(named=True):
    rounds_by_m.setdefault(r["match_key"], {}).setdefault(r["event_type"], []).append(r["round"])

RD = pl.read_parquet(OUT / "rounds.parquet").select(["match_key", "map_name", "day"]).unique("match_key")
meta = {r["match_key"]: (r["map_name"], int(r["day"])) for r in RD.iter_rows(named=True)}

# D8 signature and complete/classes come from the scratchpad table; rebuild the minimal pieces here
R = pl.read_parquet(OUT / "rounds.parquet").sort(["match_key", "round"])
MAT = {}
for x in R.iter_rows(named=True):
    m = MAT.setdefault(x["match_key"], {"map": x["map_name"], "tA": int(x["tA"]), "tB": int(x["tB"]),
                                        "day": int(x["day"]), "rd": []})
    m["rd"].append(x)


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


def complete(m):
    n = len(m["rd"])
    return n == m["tA"] + m["tB"] and n <= 24 and [x["round"] for x in m["rd"]] == list(range(1, n + 1))


say("\nAnnouncement rounds by D8 signature (days 12-25, all matches):")
groups = collections.defaultdict(list)
for k, m in MAT.items():
    if m["day"] < 12:
        continue
    s = d8_sig(m)
    lh = sorted(rounds_by_m.get(k, {}).get("round_announce_last_round_half", []))
    mp = sorted(rounds_by_m.get(k, {}).get("round_announce_match_point", []))
    groups[str(s)].append((lh, mp, m["map"]))
for s in sorted(groups, key=lambda z: -len(groups[z])):
    lhc = collections.Counter(tuple(lh) for lh, _, _ in groups[s])
    say(f"  sig {s:<10} matches={len(groups[s]):>5}  last_round_half rounds (top 5): {lhc.most_common(5)}")

say("\nAnnouncement rounds for complete matches, by D4 class proxy (D8 sig):")
cc = collections.defaultdict(list)
for k, m in MAT.items():
    if m["day"] >= 12 and complete(m):
        cc[str(d8_sig(m))].append(tuple(sorted(rounds_by_m.get(k, {}).get("round_announce_last_round_half", []))))
for s in cc:
    say(f"  complete sig {s:<10} n={len(cc[s]):>4} last_round_half patterns: {collections.Counter(cc[s]).most_common(4)}")

say("\nRound-announcement presence on (8,) matches for round 8 vs round 12 (all days, complete or not):")
s8 = [k for k, m in MAT.items() if d8_sig(m) == (8,)]
c8 = sum(1 for k in s8 if 8 in rounds_by_m.get(k, {}).get("round_announce_last_round_half", []))
c12 = sum(1 for k in s8 if 12 in rounds_by_m.get(k, {}).get("round_announce_last_round_half", []))
say(f"  (8,) matches n={len(s8)}: last_round_half at round 8 = {c8}; at round 12 = {c12}")
s12 = [k for k, m in MAT.items() if d8_sig(m) == (12,) and m["day"] >= 12]
c12b = sum(1 for k in s12 if 12 in rounds_by_m.get(k, {}).get("round_announce_last_round_half", []))
say(f"  (12,) matches days 12-25 n={len(s12)}: last_round_half at round 12 = {c12b}")
LOGF.close()
print("EVENTS DONE")
