"""A13: score-only test for matches without a clean player_info split (2026 tome).

ASSUMPTION (schedule): sides swap after round r when r == max_rounds // 2, or in overtime when
(r - max_rounds - 3) % 6 == 0. This is the rule in scripts/model/bakeoff.py (swap_expected). It is
checked here against player moves where player_info allows, and used only for matches that
cannot be checked that way.

For each scheduled-swap boundary (r -> r+1) with winner w(r) (side code) and freeze scores:
  swap pattern   : (t, ct)(r+1) == (ct(r) + [w==3], t(r) + [w==2])   fields follow the sides
  noswap pattern : (t, ct)(r+1) == (t(r) + [w==2], ct(r) + [w==3])   fields follow a team
A team-attached field would show noswap at every scheduled boundary. Count unambiguous noswap
boundaries (zero is expected if fields are side-attached).

Read-only. Output: stdout only (captured to logs/swarm2/A13_score_schedule.log).
"""
import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path("G:/pureskill/csds-cs2")
TOME = ROOT / "tomes/2026-10-01_to_08/tome/csds"
SUFFIX = ".2026-09-18,2026-10-08"


def read(channel, cols):
    folder = TOME / ("header" if channel == "header" else f"basic_{channel}{SUFFIX}")
    files = sorted(glob.glob(str(folder / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


def swap_expected(r, max_rounds):
    return r == max_rounds // 2 or (r > max_rounds and (r - max_rounds - 3) % 6 == 0)


hdr = read("header", ["key", "max_rounds"])
MR = {r["key"]: r["max_rounds"] for r in hdr.iter_rows(named=True)}
RE = read("round_end", ["match_key", "round", "winner_team_code"])
RS = read("round_state", ["match_key", "round", "event_type", "t_score", "ct_score"])
PI = read("player_info", ["match_key", "round", "player_id", "team_code"])

win = {(x["match_key"], int(x["round"])): x["winner_team_code"] for x in RE.iter_rows(named=True)}
frz = {}
for x in RS.filter(pl.col("event_type") == "round_freeze_end").iter_rows(named=True):
    k = (x["match_key"], int(x["round"]))
    if k not in frz and x["t_score"] is not None and x["ct_score"] is not None:
        frz[k] = (int(x["t_score"]), int(x["ct_score"]))
codes = collections.defaultdict(dict)
for x in PI.iter_rows(named=True):
    if x["team_code"] in (2, 3):
        codes[(x["match_key"], int(x["round"]))][x["player_id"]] = x["team_code"]


def clean_split(m, r):
    c = codes.get((m, r), {})
    return len(c) == 10 and sum(1 for v in c.values() if v == 2) == 5


def moved(m, r):
    """True if the players swapped sides between r and r+1 (needs both clean splits)."""
    if not (clean_split(m, r) and clean_split(m, r + 1)):
        return None
    a, b = codes[(m, r)], codes[(m, r + 1)]
    common = set(a) & set(b)
    ch = sum(1 for p in common if a[p] != b[p])
    if len(common) == 10 and ch == 10:
        return True
    if len(common) == 10 and ch == 0:
        return False
    return None


sched = collections.Counter()
per_match = collections.defaultdict(collections.Counter)
moves_on_sched = collections.Counter()
for (m, r), (t1, c1) in frz.items():
    k2 = (m, r + 1)
    if k2 not in frz or (m, r) not in win or win[(m, r)] not in (2, 3):
        continue
    if not swap_expected(r, MR[m]):
        continue
    w = win[(m, r)]
    t2, c2 = frz[k2]
    cw, tw = int(w == 3), int(w == 2)
    sw_ok = (t2, c2) == (c1 + cw, t1 + tw)
    nosw_ok = (t2, c2) == (t1 + tw, c1 + cw)
    if sw_ok and nosw_ok:
        pat = "ambig"
    elif sw_ok:
        pat = "swap"
    elif nosw_ok:
        pat = "noswap"
    else:
        pat = "neither"
    sched[pat] += 1
    per_match[m][pat] += 1
    mv = moved(m, r)
    moves_on_sched[(pat, "moved" if mv else ("stayed" if mv is False else "unknown"))] += 1

print("== A13 score-only schedule test (2026 tome) ==")
print("ASSUMPTION: scheduled swap = bakeoff.swap_expected (halftime max_rounds//2; OT after 3rd round of each 6)")
print(f"scheduled-swap boundaries (both freeze rows present, winner known): {sum(sched.values())}")
print(f"  pattern counts: {dict(sched)}")
print("  pattern x player move (moved / stayed / unknown):")
for k, v in sorted(moves_on_sched.items()):
    print(f"    {k}: {v}")

keys = list(MR)


def day(m):
    return m.split("/")[3]


def classify(m):
    c = per_match.get(m, collections.Counter())
    if c.get("noswap", 0) > 0 and c.get("swap", 0) == 0:
        return "team_pattern_only"
    if c.get("swap", 0) > 0 and c.get("noswap", 0) == 0:
        return "side_pattern_only"
    if c.get("swap", 0) > 0 and c.get("noswap", 0) > 0:
        return "mixed"
    if c.get("neither", 0) > 0:
        return "neither_present"
    if sum(c.values()) == 0:
        return "no_scheduled_boundary"
    return "ambig_only"


noclean = {m for m in keys if not any(clean_split(m, r) for r in range(1, 30))}
print(f"\nmatches with no clean 5v5 player_info round at all: {len(noclean)}")
for label, subset in [("all", keys),
                      ("days 01-02", [m for m in keys if day(m) in ("01", "02")]),
                      ("days 03-10", [m for m in keys if day(m) not in ("01", "02")])]:
    cc = collections.Counter(classify(m) for m in subset)
    print(f"{label}: n={len(subset)} {dict(cc)}")
print("\nno clean 5v5 matches (PI-less), score-only class:")
for label, subset in [("all PI-less", list(noclean)),
                      ("PI-less, days 01-02", [m for m in noclean if day(m) in ("01", "02")])]:
    cc = collections.Counter(classify(m) for m in subset)
    print(f"  {label}: n={len(subset)} {dict(cc)}")
team_like = [m for m in keys if classify(m) == "team_pattern_only"]
print(f"\nteam_pattern_only matches (score-only): {len(team_like)}; by day {dict(collections.Counter(day(m) for m in team_like))}")
print(f"neither_present matches: {sum(1 for m in keys if classify(m) == 'neither_present')}")
