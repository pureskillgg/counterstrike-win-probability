"""A13 diagnostics: why are some 2026 matches undetermined or insufficient in the whole-match test?

Reads the same 2026 tome pages as A13_score_attach_full.py (projected columns only). Reports, for
days 01-02 and for all days, the reason a match could not be evaluated past the halftime boundary.
Read-only. Output: stdout only (captured to logs/swarm2/A13_score_diag.log).
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


hdr = read("header", ["key", "max_rounds"])
MR = {r["key"]: r["max_rounds"] for r in hdr.iter_rows(named=True)}
RE = read("round_end", ["match_key", "round", "winner_team_code"])
RS = read("round_state", ["match_key", "round", "event_type", "t_score", "ct_score"])
PI = read("player_info", ["match_key", "round", "player_id", "team_code"])

win = {(x["match_key"], int(x["round"])): x["winner_team_code"] for x in RE.iter_rows(named=True)}
frz = {}
for x in RS.filter(pl.col("event_type") == "round_freeze_end").iter_rows(named=True):
    k = (x["match_key"], int(x["round"]))
    if k not in frz:
        frz[k] = (x["t_score"], x["ct_score"])
codes = collections.defaultdict(dict)
for x in PI.iter_rows(named=True):
    if x["team_code"] in (2, 3):
        codes[(x["match_key"], int(x["round"]))][x["player_id"]] = x["team_code"]
pi_rows = collections.Counter()
for x in PI.iter_rows(named=True):
    pi_rows[(x["match_key"], int(x["round"]))] += 1

rounds_by_match = collections.defaultdict(set)
for (m, r) in set(win) | set(frz) | set(codes):
    rounds_by_match[m].add(r)

reason = {}
max_round = {}
for m in MR:
    rs = sorted(rounds_by_match.get(m, ()))
    max_round[m] = max(rs) if rs else 0
    # clean start round
    x_set = None
    for r in rs:
        c = codes.get((m, r), {})
        if len(c) == 10 and sum(1 for v in c.values() if v == 2) == 5:
            x_set = {p for p, v in c.items() if v == 2}
            break
    if x_set is None:
        reason[m] = "no clean 5v5 player_info round"
        continue

    def side(r):
        c = codes.get((m, r), {})
        if len(c) != 10:
            return None
        xs = [c.get(p) for p in x_set]
        if any(v is None for v in xs):
            return None
        if all(v == 2 for v in xs):
            return 2
        if all(v == 3 for v in xs):
            return 3
        return None

    first_break = None
    n_eval = 0
    prefix_ok = True
    for r in range(1, max_round[m] + 1):
        sx = side(r)
        if (m, r) in frz and prefix_ok and sx is not None:
            n_eval += 1
        w = win.get((m, r))
        if prefix_ok and w in (2, 3) and sx is not None:
            pass
        else:
            prefix_ok = False
            if first_break is None:
                if w not in (2, 3):
                    cause = "round_end winner missing or null"
                elif sx is None:
                    cause = f"player_info not a clean 5v5 (rows={pi_rows.get((m, r), 0)})"
                else:
                    cause = "other"
                first_break = (r, cause)
    if n_eval == 0:
        reason[m] = f"prefix break before any evaluable round: round {first_break[0]} {first_break[1]}" if first_break else "no freeze rows"
    elif first_break is not None:
        reason[m] = f"evaluated {n_eval} rounds; prefix broke at round {first_break[0]}: {first_break[1]}"
    else:
        reason[m] = "evaluated all rounds"


def day(m):
    return m.split("/")[3]


def show(label, keys):
    print(f"\n{label}: n={len(keys)}")
    buckets = collections.Counter()
    for m in keys:
        s = reason.get(m, "evaluated all rounds")
        if s.startswith("evaluated"):
            buckets["evaluated (no break)"] += 1
        elif s.startswith("no clean"):
            buckets["no clean 5v5 round"] += 1
        elif "player_info not a clean" in s:
            buckets["prefix broke: player_info not 5v5"] += 1
        elif "winner missing" in s:
            buckets["prefix broke: winner missing"] += 1
        elif "no freeze" in s:
            buckets["no freeze rows"] += 1
        else:
            buckets["other: " + s[:60]] += 1
    for k, v in buckets.most_common():
        print(f"  {k}: {v}")
    short = [m for m in keys if max_round.get(m, 0) < 13]
    print(f"  matches whose highest round is < 13 (no halftime possible at MR24): {len(short)}")


all_keys = list(MR)
show("all 2026 matches", all_keys)
show("days 01-02", [m for m in all_keys if day(m) in ("01", "02")])
show("days 03-10", [m for m in all_keys if day(m) not in ("01", "02")])

halftime = [m for m in all_keys if max_round[m] >= 13 and MR[m] == 24 or max_round[m] >= 9 and MR[m] == 16]
show("matches reaching halftime (max_round >= 13 for MR24, >= 9 for MR16)", halftime)
ht12 = [m for m in halftime if day(m) in ("01", "02")]
show("days 01-02 reaching halftime", ht12)

print("\nSample reasons (days 01-02, first 8):")
for m in [m for m in all_keys if day(m) in ("01", "02")][:8]:
    print(f"  {m[:40]} MR{MR[m]} maxr={max_round[m]}: {reason.get(m, 'evaluated all rounds')}")
