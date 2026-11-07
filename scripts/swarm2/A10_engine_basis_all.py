"""A10 engine-basis agreement for all 150 sampled matches (read-only for E:; writes under logs\\swarm2).

Engine basis (inferred, see A10 report): alive set at freeze end = roster players (player_info, round)
present in raw player_status at the freeze-end tick, by roster team; alive count at a snapshot =
basis size minus distinct in-window victims with death tick <= snapshot tick. Compared with the
1 Hz table counts per snapshot (data\\state_sample\\players_1hz.parquet). Death-based agreement (the
validation's rule) is recomputed alongside for the same snapshots.
"""
import collections
import json
import os
import sys
import time

import polars as pl
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"G:\pureskill\csds-cs2"
E_CSDS = r"E:\csds_final\csds"
OUT_JSON = os.path.join(ROOT, "logs", "swarm2", "A10_engine_basis_all.json")
T, CT, START, TICK = 2, 3, 5, 64

tbl = pl.read_parquet(os.path.join(ROOT, "data", "state_sample", "players_1hz.parquet"))
cnt = tbl.group_by(["match_key", "round", "tick"]).agg(
    (pl.col("team_code") == T).sum().alias("tT"), (pl.col("team_code") == CT).sum().alias("tCT"))
rounds = pl.read_parquet(os.path.join(ROOT, "data", "state_sample", "rounds.parquet"))
matches = sorted(rounds["match_key"].unique().to_list())


def rd(base, ch, cols):
    path = os.path.join(base, ch)
    names = pq.ParquetDataset(path).schema.names
    return pl.from_arrow(pq.read_table(path, columns=[c for c in cols if c in names]))


t0 = time.perf_counter()
n_snap = n_eng_ok = n_death_ok = 0
eng_bad = []
death_bad = 0
disagree_side = 0
for mk in matches:
    p = mk.split("/")
    base = os.path.join(E_CSDS, p[1], p[2], p[3], p[4])
    rs = rd(base, "round_state", ["round", "tick", "event_type"]).filter(pl.col("event_type") == "round_freeze_end")
    fz_map = {r: int(t) for r, t in rs.group_by("round").agg(pl.col("tick").min()).iter_rows()}
    end_df = rd(base, "round_end", ["round", "tick"])
    end_map = {r: int(t) for r, t in end_df.group_by("round").agg(pl.col("tick").min()).iter_rows()}
    pi = rd(base, "player_info", ["round", "player_id", "team_code"])
    roster = collections.defaultdict(dict)
    for r, pid, team in pi.select(["round", "player_id", "team_code"]).iter_rows():
        roster[r][pid] = team
    st = rd(base, "player_status", ["round", "tick", "player_id"])
    de = rd(base, "player_death", ["round", "tick", "player_id", "player_team_code"]).sort("tick")
    # alive set at freeze end
    basis = {}
    for r, fz in fz_map.items():
        at = st.filter((pl.col("round") == r) & (pl.col("tick") == fz))["player_id"].to_list()
        basis[r] = {pid: roster[r][pid] for pid in at if pid in roster[r]}
    # in-window deaths, first per victim
    dd = collections.defaultdict(list)
    for r, tick, pid, team in de.select(["round", "tick", "player_id", "player_team_code"]).iter_rows():
        if r not in fz_map or tick < fz_map[r] or tick > end_map.get(r, -1) or r not in roster:
            continue
        if any(v[0] == pid for v in dd[r]):
            continue
        side_roster = roster[r].get(pid)
        if team is not None and side_roster is not None and team != side_roster:
            disagree_side += 1
        dd[r].append((tick, pid, side_roster if side_roster is not None else team))
    sub = cnt.filter(pl.col("match_key") == mk)
    for r, t, tT, tCT in sub.select(["round", "tick", "tT", "tCT"]).iter_rows():
        if r not in fz_map:
            continue
        n_snap += 1
        bT = sum(1 for pid, s in basis[r].items() if s == T)
        bCT = sum(1 for pid, s in basis[r].items() if s == CT)
        dT = sum(1 for tick, pid, s in dd[r] if tick <= t and s == T)
        dCT = sum(1 for tick, pid, s in dd[r] if tick <= t and s == CT)
        eT, eCT = bT - dT, bCT - dCT
        if (tT, tCT) == (eT, eCT):
            n_eng_ok += 1
        else:
            eng_bad.append([mk, r, t, tT, tCT, eT, eCT])
        if (tT, tCT) == (START - sum(1 for tick, pid, s in dd[r] if tick <= t and s == T),
                         START - sum(1 for tick, pid, s in dd[r] if tick <= t and s == CT)):
            n_death_ok += 1
        else:
            death_bad += 1

eng_rounds = collections.Counter((x[0], x[1]) for x in eng_bad)
res = {"snapshots": n_snap, "engine_basis_agree": n_eng_ok,
       "engine_basis_agree_share": round(n_eng_ok / n_snap, 6),
       "engine_basis_mismatch": len(eng_bad), "engine_basis_mismatch_rounds": {f"{m}|r{r}": c for (m, r), c in eng_rounds.items()},
       "engine_basis_mismatch_T_minus_engine": sorted({x[3] - x[5] for x in eng_bad}),
       "engine_basis_mismatch_CT_minus_engine": sorted({x[4] - x[6] for x in eng_bad}),
       "death_based_agree": n_death_ok, "death_based_mismatch": death_bad,
       "death_roster_team_vs_death_team_disagree": disagree_side,
       "wall_s": round(time.perf_counter() - t0, 1)}
print(json.dumps(res, indent=1))
with open(OUT_JSON, "w", encoding="utf-8") as fh:
    json.dump({**res, "engine_basis_mismatch_first50": eng_bad[:50]}, fh, indent=1, default=str)
print("written", OUT_JSON)
