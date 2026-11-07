"""A10 impact checks (read-only; writes only under logs\\swarm2).

I1  bake-off states in the six absent-at-start rounds: count states per round (1 + in-window
    unique deaths, as build_states builds them) and the total for the kept set.
I2  mid-round connect rule across the full kept set: a non-bot player_connect event for a roster
    player with freeze_end < tick < round_end. Compare with the six presence cases.
I3  buy-time offset: round_state buytime_ended minus round_freeze_end, in seconds, over the 150
    sampled matches (clean rounds from rounds.parquet).
"""
import collections
import importlib.util
import json
import os
import sys

import polars as pl
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"G:\pureskill\csds-cs2"
E_CSDS = r"E:\csds_final\csds"
OUT_JSON = os.path.join(ROOT, "logs", "swarm2", "A10_impact_checks.json")
T, CT, START, TICK = 2, 3, 5, 64

spec = importlib.util.spec_from_file_location("extract_1hz", os.path.join(ROOT, "scripts", "eda", "extract_state_1hz.py"))
ex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex)
fv = ex.fv
kept = ex.bakeoff_kept()
rounds_ok = kept["rounds_ok"]
live, rd, deaths, players = kept["live"], kept["rd"], kept["deaths"], kept["players"]

fs = json.load(open(os.path.join(ROOT, "logs", "swarm2", "A10_fullset_checks.json"), encoding="utf-8"))
six = {(c[0], c[1]) for c in fs["D_live_start_presence"]["rounds_presence_not_5v5"]}

# I1: states per round, mirroring build_states (live start state + one per unique in-window victim)
states_total = 0
six_states = {}
for k in rounds_ok:
    fz, end = live[k]["tick"], rd[k]["end"]
    pl_map = players.get(k, {})
    seen = set()
    n_win = 0
    for d in deaths.get(k, []):
        if d["tick"] < fz or d["tick"] > end or d["pid"] in seen:
            continue
        side = d["team"] if d["team"] is not None else pl_map.get(d["pid"])
        if side not in (T, CT):
            continue
        seen.add(d["pid"])
        n_win += 1
    states_total += 1 + n_win
    if k in {(m, r) for m, r in six}:
        six_states[f"{k[0]}|r{k[1]}"] = 1 + n_win
I1 = {"states_total_recomputed": states_total, "states_bakeoff_log": 234265,
      "six_rounds_states": six_states, "six_rounds_total_states": sum(six_states.values()),
      "share_of_states": round(sum(six_states.values()) / states_total, 6)}
print("== I1 bake-off states in the six rounds ==")
print(json.dumps(I1, indent=1))

# I2: mid-round connect rule over the full kept set
pc = fv.read("player_connect", ["match_key", "round", "tick", "player_id", "is_bot"])
print("\n== I2 mid-round connect rule ==")
print(f"player_connect rows (all tome matches): {pc.height}; is_bot True: "
      f"{pc.filter(pl.col('is_bot') == True).height}")
pc_nb = pc.filter(pl.col("is_bot") == False)
conn_hits = []
for x in pc_nb.iter_rows(named=True):
    k = (x["match_key"], int(x["round"]))
    if k not in live:
        continue
    fz, end = live[k]["tick"], rd[k]["end"]
    if fz < int(x["tick"]) < end:
        roster = players.get(k, {})
        conn_hits.append({"k": k, "pid": x["player_id"], "tick": int(x["tick"]), "in_roster": x["player_id"] in roster,
                          "roster_team": roster.get(x["player_id"]), "k_s": (int(x["tick"]) - fz) / TICK})
rounds_connect = {h["k"] for h in conn_hits}
rounds_connect_in_roster = {h["k"] for h in conn_hits if h["in_roster"]}
I2 = {"kept_rounds_with_midround_nonbot_connect": len(rounds_connect),
      "of_which_pid_in_roster": len(rounds_connect_in_roster),
      "six_presence_rounds_detected_by_connect": sorted(f"{m}|r{r}" for (m, r) in six if (m, r) in rounds_connect),
      "six_presence_rounds_NOT_detected_by_connect": sorted(f"{m}|r{r}" for (m, r) in six if (m, r) not in rounds_connect),
      "connect_rounds_not_in_six": sorted(f"{m}|r{r}" for (m, r) in rounds_connect if (m, r) not in six),
      "connect_hit_list_first_20": [{**h, "k": f"{h['k'][0]}|r{h['k'][1]}"} for h in conn_hits[:20]]}
print(json.dumps(I2, indent=1, default=str))

# I3: buy-time offset in the 150 sampled matches
rounds = pl.read_parquet(os.path.join(ROOT, "data", "state_sample", "rounds.parquet"))
offs = []
for mk in sorted(rounds["match_key"].unique().to_list()):
    p = mk.split("/")
    base = os.path.join(E_CSDS, p[1], p[2], p[3], p[4])
    rs = pl.from_arrow(pq.read_table(os.path.join(base, "round_state"), columns=["round", "tick", "event_type"]))
    fz = rs.filter(pl.col("event_type") == "round_freeze_end").group_by("round").agg(pl.col("tick").min().alias("fz"))
    bt = rs.filter(pl.col("event_type") == "buytime_ended").group_by("round").agg(pl.col("tick").min().alias("bt"))
    j = fz.join(bt, on="round", how="inner")
    rr = rounds.filter(pl.col("match_key") == mk).select("round")
    j = j.join(rr, on="round", how="inner")
    for r, f, b in j.select(["round", "fz", "bt"]).iter_rows():
        offs.append((b - f) / TICK)
import statistics
if offs:
    offs_sorted = sorted(offs)
    I3 = {"rounds": len(offs), "min_s": round(min(offs), 3), "max_s": round(max(offs), 3),
          "median_s": round(statistics.median(offs), 3),
          "p10_s": round(offs_sorted[int(0.1 * len(offs))], 3), "p90_s": round(offs_sorted[int(0.9 * len(offs))], 3),
          "rounds_without_buytime_row": int(rounds.height - len(offs))}
else:
    I3 = {"rounds": 0}
print("\n== I3 buy-time offset (buytime_ended - round_freeze_end, s) ==")
print(json.dumps(I3, indent=1))

res = {"I1_states": I1, "I2_connect_rule": I2, "I3_buytime": I3}
with open(OUT_JSON, "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=1, default=str)
print(f"\nwritten {OUT_JSON}")
