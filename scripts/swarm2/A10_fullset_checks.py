"""A10 full kept-set checks for the round-16 pattern (read-only; writes only under logs\\swarm2).

Uses the bake-off kept set (reproduced by extract_state_1hz.bakeoff_kept, which asserts the
bake-off counts). Two checks, both over all kept 2026 rounds (28,683 expected):

C  elimination-end consistency from the event channels only: for rounds whose win_reason_code
   is 8 (CT win) or 9 (T win), the losing side should have 5 distinct in-window victims.
D  live-start presence: every roster player of the round (player_info, via bake-off players
   map) should appear in raw player_status at the freeze-end tick. Reads player_status
   column-projected, one match at a time.
"""
import collections
import importlib.util
import json
import os
import sys
import time

import polars as pl
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"G:\pureskill\csds-cs2"
OUT_JSON = os.path.join(ROOT, "logs", "swarm2", "A10_fullset_checks.json")
T, CT, START = 2, 3, 5

spec = importlib.util.spec_from_file_location("extract_1hz", os.path.join(ROOT, "scripts", "eda", "extract_state_1hz.py"))
ex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex)
fv = ex.fv

t0 = time.perf_counter()
kept = ex.bakeoff_kept()
rounds_ok = kept["rounds_ok"]
live, rd, deaths, players = kept["live"], kept["rd"], kept["deaths"], kept["players"]
print(f"kept rounds: {len(rounds_ok)}; kept matches: {len({k[0] for k in rounds_ok})}; "
      f"t={time.perf_counter() - t0:.1f}s")

RE = fv.read("round_end", ["match_key", "round", "win_reason_code", "winner_team_code"])
reason = {(x["match_key"], int(x["round"])): int(x["win_reason_code"]) for x in RE.iter_rows(named=True)}

# ---------------- C: elimination-end consistency over the full kept set ----------------
C = {"kept_rounds": len(rounds_ok), "by_reason_and_losing_victims": {}, "elim_rounds": 0,
     "elim_losing_victims_ne5": [], "non_elim_losing_victims_eq5": collections.Counter(),
     "winner_reason_inconsistent": 0}
tab = collections.Counter()
for k in rounds_ok:
    m, r = k
    fz, end = live[k]["tick"], rd[k]["end"]
    winner = rd[k]["winner"]
    pl_map = players.get(k, {})
    seen, tT, tCT = set(), set(), set()
    for d in deaths.get(k, []):
        if d["tick"] < fz or d["tick"] > end or d["pid"] in seen:
            continue
        seen.add(d["pid"])
        side = d["team"] if d["team"] is not None else pl_map.get(d["pid"])
        if side == T:
            tT.add(d["pid"])
        elif side == CT:
            tCT.add(d["pid"])
    rc = reason.get(k)
    lose = CT if winner == T else (T if winner == CT else None)
    n_lose = len(tT) if lose == T else (len(tCT) if lose == CT else -1)
    tab[(rc, n_lose)] += 1
    if rc in (8, 9):
        C["elim_rounds"] += 1
        if (rc == 8 and winner != CT) or (rc == 9 and winner != T):
            C["winner_reason_inconsistent"] += 1
        if n_lose != START:
            C["elim_losing_victims_ne5"].append([m, r, rc, n_lose, lose])
    elif n_lose == START:
        C["non_elim_losing_victims_eq5"][rc] += 1
C["by_reason_and_losing_victims"] = {f"code {rc} / losing victims {n}": c for (rc, n), c in sorted(
    tab.items(), key=lambda x: (str(x[0][0]), x[0][1]))}
C["non_elim_losing_victims_eq5"] = dict(C["non_elim_losing_victims_eq5"])
print("\n== C: elimination consistency (full kept set) ==")
print(f"  rounds: {C['kept_rounds']}; elimination-end rounds (code 8/9): {C['elim_rounds']}")
print(f"  code/losing-victims table: {C['by_reason_and_losing_victims']}")
print(f"  elimination rounds with losing victims != 5: {len(C['elim_losing_victims_ne5'])} -> "
      f"{C['elim_losing_victims_ne5'][:10]}")
print(f"  winner vs code inconsistent: {C['winner_reason_inconsistent']}")
print(f"  non-elimination rounds with 5 losing victims, by code: {C['non_elim_losing_victims_eq5']}")

# ---------------- D: live-start presence of the roster, full kept set (E: player_status) ----------------
D = {"rounds": len(rounds_ok), "rounds_roster_not_5v5": 0, "rounds_presence_not_5v5": [],
     "rounds_roster_ne_5v5": [], "extra_present_not_in_roster": [], "matches_read": 0,
     "status_read_s": 0.0}
by_match = collections.defaultdict(list)
for k in rounds_ok:
    by_match[k[0]].append(k)
print("\n== D: live-start presence (E: player_status at freeze-end tick) ==")
t1 = time.perf_counter()
for i, (m, ks) in enumerate(sorted(by_match.items())):
    # roster check from the bake-off players map
    fz_df = pl.DataFrame({"round": [k[1] for k in ks], "tick": [live[k]["tick"] for k in ks]},
                         schema={"round": pl.Int64, "tick": pl.Int64})
    path = os.path.join(ex.e_dir(m), "player_status")
    ta = time.perf_counter()
    names = pq.ParquetDataset(path).schema.names
    st = pl.from_arrow(pq.read_table(path, columns=["round", "tick", "player_id"]))
    D["status_read_s"] += time.perf_counter() - ta
    D["matches_read"] += 1
    st = st.with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64))
    at = st.join(fz_df, on=["round", "tick"], how="inner").select(["round", "player_id"])
    present = collections.defaultdict(set)
    for r, pid in at.iter_rows():
        present[r].add(pid)
    for k in ks:
        r = k[1]
        roster = players.get(k, {})
        rt = {pid for pid, team in roster.items() if team == T}
        rct = {pid for pid, team in roster.items() if team == CT}
        if not (len(rt) == START and len(rct) == START):
            D["rounds_roster_ne_5v5"].append([m, r, len(rt), len(rct)])
        p = present.get(r, set())
        pt, pct = len(p & rt), len(p & rct)
        extra = sorted(p - set(roster))
        if extra:
            D["extra_present_not_in_roster"].append([m, r, extra])
        if pt != START or pct != START:
            D["rounds_presence_not_5v5"].append([m, r, pt, pct, sorted(rt - p), sorted(rct - p),
                                                 int(rd[k]["end"] - live[k]["tick"])])
    if (i + 1) % 300 == 0:
        print(f"  {i + 1}/{len(by_match)} matches read; {time.perf_counter() - t1:.1f}s")
D["rounds_roster_not_5v5"] = len(D["rounds_roster_ne_5v5"])
D["n_rounds_presence_not_5v5"] = len(D["rounds_presence_not_5v5"])
D["by_side_missing"] = dict(collections.Counter(
    ("T" if len(x[4]) > 0 else "") + ("CT" if len(x[5]) > 0 else "") for x in D["rounds_presence_not_5v5"]))
print(f"  roster not 5v5 (should be 0): {D['rounds_roster_not_5v5']}")
print(f"  rounds with live-start presence not 5v5: {D['n_rounds_presence_not_5v5']}")
print(f"  first: {D['rounds_presence_not_5v5'][:15]}")
print(f"  extra present not in roster: {len(D['extra_present_not_in_roster'])} "
      f"{D['extra_present_not_in_roster'][:5]}")
print(f"  status read total {D['status_read_s']:.1f}s over {D['matches_read']} matches")

res = {"C_elimination": C, "D_live_start_presence": D, "wall_s": round(time.perf_counter() - t0, 1)}
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=1, default=str)
print(f"\nwritten {OUT_JSON}; wall {time.perf_counter() - t0:.1f}s")
