"""A10 round-16 snapshot comparison (read-only; writes only under logs\\swarm2).

Engine basis for the round: alive set at freeze end = players present in raw player_status at the
freeze-end tick (roster players absent at freeze end are not counted). Alive count at a snapshot =
basis size minus distinct in-window victims with death tick <= snapshot tick. Compared with the
1 Hz table counts (data\\state_sample\\players_1hz.parquet) and with the death-based count (5 minus
deaths) that the validation uses. Also the bake-off states for the same round.
"""
import json
import os
import sys

import polars as pl
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"G:\pureskill\csds-cs2"
E_CSDS = r"E:\csds_final\csds"
MK = "csds/2026/10/07/22e9c3e7-a333-4ebc-a51c-19ebbcc13ef1/csds"
ROUND = 16
TICK = 64
T, CT = 2, 3
OUT_JSON = os.path.join(ROOT, "logs", "swarm2", "A10_r16_engine_basis.json")

base = os.path.join(E_CSDS, "2026", "10", "07", "22e9c3e7-a333-4ebc-a51c-19ebbcc13ef1")


def rd(ch, cols):
    names = pq.ParquetDataset(os.path.join(base, ch)).schema.names
    return pl.from_arrow(pq.read_table(os.path.join(base, ch), columns=[c for c in cols if c in names]))


rs = rd("round_state", ["round", "tick", "event_type"]).filter(pl.col("round") == ROUND)
fz = int(rs.filter(pl.col("event_type") == "round_freeze_end")["tick"].min())
re_ = rd("round_end", ["round", "tick"]).filter(pl.col("round") == ROUND)
end = int(re_["tick"][0])
pi = rd("player_info", ["round", "player_id", "team_code"]).filter(pl.col("round") == ROUND)
roster = {pid: team for pid, team in pi.select(["player_id", "team_code"]).iter_rows()}
st = rd("player_status", ["round", "tick", "player_id"]).filter((pl.col("round") == ROUND) & (pl.col("tick") == fz))
basis = {pid: roster[pid] for pid in st["player_id"].to_list() if pid in roster}
de = rd("player_death", ["round", "tick", "player_id", "player_team_code"]).filter(pl.col("round") == ROUND).sort("tick")
deaths = []
seen = set()
for tick, pid in de.select(["tick", "player_id"]).iter_rows():
    if tick < fz or tick > end or pid in seen:
        continue
    seen.add(pid)
    deaths.append((tick, pid))
print(f"fz {fz} end {end}; roster {len(roster)}; present at fz {len(basis)}; basis T {sorted(p for p, t in basis.items() if t == T)} "
      f"CT {sorted(p for p, t in basis.items() if t == CT)}")
print(f"in-window deaths (tick, victim): {deaths}")

tbl = pl.read_parquet(os.path.join(ROOT, "data", "state_sample", "players_1hz.parquet")).filter(
    (pl.col("match_key") == MK) & (pl.col("round") == ROUND))
cnt = tbl.group_by("tick").agg((pl.col("team_code") == T).sum().alias("tT"),
                                (pl.col("team_code") == CT).sum().alias("tCT")).sort("tick")
tmap = {t: (a, b) for t, a, b in cnt.iter_rows()}

rows = []
n_engine_mismatch = 0
n_death_mismatch = 0
bad_engine = []
for k in range(0, (end - fz + TICK - 1) // TICK):
    t = fz + TICK * k
    dead_before = {pid for tick, pid in deaths if tick <= t}
    eT = sum(1 for p, s in basis.items() if s == T and p not in dead_before)
    eCT = sum(1 for p, s in basis.items() if s == CT and p not in dead_before)
    dT = 5 - sum(1 for tick, pid in deaths if tick <= t and roster.get(pid) == T)
    dCT = 5 - sum(1 for tick, pid in deaths if tick <= t and roster.get(pid) == CT)
    tT, tCT = tmap.get(t, (None, None))
    rows.append({"k": k, "tick": t, "table_T": tT, "table_CT": tCT, "engine_T": eT, "engine_CT": eCT,
                 "death_T": dT, "death_CT": dCT})
    if (tT, tCT) != (eT, eCT):
        n_engine_mismatch += 1
        bad_engine.append([k, tT, tCT, eT, eCT])
    if (tT, tCT) != (dT, dCT):
        n_death_mismatch += 1
print(f"snapshots {len(rows)}; table != engine basis: {n_engine_mismatch}; table != death-based: {n_death_mismatch}")
print(f"table != engine basis, k list: {[r[0] for r in bad_engine]}")
print(f"  rows (k, table_T, table_CT, engine_T, engine_CT): {[tuple(r) for r in bad_engine[:3]]} ... {[tuple(r) for r in bad_engine[-2:]]}")
diff_T = {r[1] - r[3] for r in bad_engine}
diff_CT = {r[2] - r[4] for r in bad_engine}
print(f"  table minus engine: T values {diff_T}, CT values {diff_CT}")
k_eq = [r["k"] for r in rows if r["table_T"] == r["engine_T"] and r["table_CT"] == r["engine_CT"]]
print(f"  snapshots where table equals engine basis: {len(k_eq)} (k {k_eq[:3]}...{k_eq[-3:] if k_eq else ''})")
p4_rows = tbl.filter(pl.col("player_id") == 4).height
print(f"1 Hz table rows for player 4 in this round: {p4_rows}")

# bake-off states for this round (build_states logic: start 5/5, one state per unique in-window victim)
bo_T, bo_CT = 5, 5
bo_states = [("live_start", bo_T, bo_CT)]
for tick, pid in deaths:
    side = roster[pid]
    if side == T:
        bo_T -= 1
    else:
        bo_CT -= 1
    bo_states.append((f"death {pid} @ {tick}", bo_T, bo_CT))
eng_states = []
eT0 = sum(1 for p, s in basis.items() if s == T)
eC0 = sum(1 for p, s in basis.items() if s == CT)
eT_, eC_ = eT0, eC0
eng_states.append(("live_start", eT_, eC_))
for tick, pid in deaths:
    if roster[pid] == T:
        eT_ -= 1
    else:
        eC_ -= 1
    eng_states.append((f"death {pid} @ {tick}", eT_, eC_))
print("bake-off states (label, alive_T, alive_CT):", bo_states)
print("engine-basis states (label, alive_T, alive_CT):", eng_states)

with open(OUT_JSON, "w", encoding="utf-8") as fh:
    json.dump({"fz": fz, "end": end, "basis_T": sorted(p for p, t in basis.items() if t == T),
               "basis_CT": sorted(p for p, t in basis.items() if t == CT), "deaths": deaths,
               "snapshots": len(rows), "table_ne_engine": n_engine_mismatch,
               "table_ne_death": n_death_mismatch, "table_ne_engine_k": [r[0] for r in bad_engine],
               "table_minus_engine_T": sorted(diff_T), "table_minus_engine_CT": sorted(diff_CT),
               "p4_rows_in_table": p4_rows, "bakeoff_states": bo_states,
               "engine_states": eng_states}, fh, indent=1, default=str)
print("written", OUT_JSON)
