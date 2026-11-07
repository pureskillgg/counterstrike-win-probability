"""A10 elimination timing and identity check (read-only; writes only under logs\\swarm2).

For every kept round with win_reason_code 8 or 9 (elimination ends, per the bake-off's
winner/reason agreement), check (1) the last losing-side in-window death equals round_end,
and (2) for the rounds where the losing side has != 5 distinct in-window victims, whether the
losing-side victims equal the losing-side players present in player_status at freeze end.
"""
import importlib.util
import json
import os
import sys

import polars as pl
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"G:\pureskill\csds-cs2"
OUT_JSON = os.path.join(ROOT, "logs", "swarm2", "A10_elim_timing.json")
T, CT = 2, 3

spec = importlib.util.spec_from_file_location("extract_1hz", os.path.join(ROOT, "scripts", "eda", "extract_state_1hz.py"))
ex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex)
fv = ex.fv
kept = ex.bakeoff_kept()
rounds_ok = kept["rounds_ok"]
live, rd, deaths, players = kept["live"], kept["rd"], kept["deaths"], kept["players"]
RE = fv.read("round_end", ["match_key", "round", "win_reason_code"])
reason = {(x["match_key"], int(x["round"])): int(x["win_reason_code"]) for x in RE.iter_rows(named=True)}

n_elim = 0
last_eq_end = 0
last_ne_end = []
exceptions = []
for k in rounds_ok:
    rc = reason.get(k)
    if rc not in (8, 9):
        continue
    n_elim += 1
    fz, end = live[k]["tick"], rd[k]["end"]
    winner = rd[k]["winner"]
    lose = T if winner == CT else CT
    pl_map = players.get(k, {})
    seen = {}
    for d in deaths.get(k, []):
        if d["tick"] < fz or d["tick"] > end or d["pid"] in seen:
            continue
        side = d["team"] if d["team"] is not None else pl_map.get(d["pid"])
        if side is None:
            continue
        seen[d["pid"]] = (d["tick"], side)
    losing = {pid: t for pid, (t, side) in seen.items() if side == lose}
    last = max(losing.values()) if losing else None
    if last == end:
        last_eq_end += 1
    else:
        last_ne_end.append([k[0], k[1], rc, last, end])
    if len(losing) != 5:
        exceptions.append({"match": k[0], "round": k[1], "reason": rc, "losing_side": lose,
                           "losing_victims": sorted(losing), "n": len(losing), "fz": fz, "end": end,
                           "roster_losing": sorted(p for p, team in pl_map.items() if team == lose)})

print(f"elimination rounds (codes 8/9): {n_elim}")
print(f"last losing-side in-window death == round_end: {last_eq_end} of {n_elim}")
print(f"  not equal: {len(last_ne_end)} -> {last_ne_end[:10]}")
print(f"losing-side victims != 5: {len(exceptions)}")

# identity check: are the losing-side victims exactly the losing-side players present at freeze end?
by_match = {}
for e in exceptions:
    by_match.setdefault(e["match"], []).append(e)
for mk, items in by_match.items():
    p = mk.split("/")
    path = os.path.join(ex.e_dir(mk), "player_status")
    st = pl.from_arrow(pq.read_table(path, columns=["round", "tick", "player_id"]))
    for e in items:
        at = st.filter((pl.col("round") == e["round"]) & (pl.col("tick") == e["fz"]))["player_id"].to_list()
        present_losing = sorted(set(at) & set(e["roster_losing"]))
        e["present_losing_at_fz"] = present_losing
        e["victims_equal_present_at_fz"] = present_losing == e["losing_victims"]
        print(f"  {mk[-45:]} r{e['round']} code {e['reason']} losing side {e['losing_side']}: "
              f"victims {e['losing_victims']}; present at freeze end {present_losing}; "
              f"equal={e['victims_equal_present_at_fz']}")

res = {"elim_rounds": n_elim, "last_loser_death_eq_round_end": last_eq_end,
       "last_loser_death_ne_round_end": last_ne_end, "exceptions": exceptions}
with open(OUT_JSON, "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=1, default=str)
print(f"written {OUT_JSON}")
