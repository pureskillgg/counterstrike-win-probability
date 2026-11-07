"""A10 probe (read-only, raw E: channels): for the six kept rounds whose roster player is absent
from player_status at freeze end, report when the absent player first appears in player_status,
whether they die in the round, their connect/spawn/disconnect/team_change events, and the
round-end reason. Input list comes from logs\\swarm2\\A10_fullset_checks.json (section D)."""
import json
import os
import sys

import polars as pl
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"G:\pureskill\csds-cs2"
E_CSDS = r"E:\csds_final\csds"
TICK = 64
T, CT = 2, 3

with open(os.path.join(ROOT, "logs", "swarm2", "A10_fullset_checks.json"), encoding="utf-8") as fh:
    fs = json.load(fh)
cases = fs["D_live_start_presence"]["rounds_presence_not_5v5"]


def e_dir(mk):
    p = mk.split("/")
    return os.path.join(E_CSDS, p[1], p[2], p[3], p[4])


def rd(mk, ch, cols=None):
    path = os.path.join(e_dir(mk), ch)
    if not os.path.exists(path):
        return None
    names = pq.ParquetDataset(path).schema.names
    use = [c for c in (cols or names) if c in names]
    return pl.from_arrow(pq.read_table(path, columns=use))


out = []
for mk, r, pt, pct, miss_t, miss_ct, span in cases:
    absent = miss_t + miss_ct
    print(f"\n### {mk} round {r}: presence T={pt} CT={pct}; absent roster ids {absent}")
    rs = rd(mk, "round_state", ["round", "tick", "event_type"])
    fz = int(rs.filter((pl.col("round") == r) & (pl.col("event_type") == "round_freeze_end"))["tick"][0])
    re_ = rd(mk, "round_end", ["round", "tick", "winner_team_code", "win_reason_code", "player_count"])
    rr = re_.filter(pl.col("round") == r).row(0, named=True)
    print(f"  freeze_end {fz}, round_end {rr['tick']} (span {rr['tick'] - fz} ticks), winner {rr['winner_team_code']}, "
          f"reason {rr['win_reason_code']}, player_count {rr['player_count']}")
    st = rd(mk, "player_status", ["round", "tick", "player_id", "health"])
    de = rd(mk, "player_death", ["round", "tick", "player_id", "player_team_code", "attacker_id"])
    pc = rd(mk, "player_connect", ["round", "tick", "player_id", "is_bot"])
    ps = rd(mk, "player_spawn", ["round", "tick", "player_id", "player_team_code"])
    dc = rd(mk, "player_disconnect", ["round", "tick", "player_id"])
    tc = rd(mk, "team_change", ["round", "tick", "player_id", "old_team_code", "new_team_code"])
    for pid in absent:
        s = st.filter((pl.col("round") == r) & (pl.col("player_id") == pid)).sort("tick")
        first = int(s["tick"][0]) if s.height else None
        print(f"  player {pid}: status rows in round {s.height}; first tick {first} "
              f"(k={(first - fz) / TICK if first is not None else 'n/a'}); last tick {int(s['tick'][-1]) if s.height else None}")
        dd = de.filter((pl.col("round") == r) & (pl.col("player_id") == pid)).select(["tick", "player_team_code", "attacker_id"]).rows()
        print(f"    deaths in round {r}: {dd}")
        for name, df in (("connect", pc), ("spawn", ps), ("disconnect", dc), ("team_change", tc)):
            if df is None:
                continue
            hit = df.filter((pl.col("player_id") == pid) & pl.col("round").is_between(r - 1, r + 1))
            if hit.height:
                print(f"    {name} rounds {r-1}-{r+1}: {hit.sort('tick').rows()}")
        out.append({"match": mk, "round": r, "absent": pid, "first_status_tick": first,
                    "deaths_in_round": dd, "freeze_end": fz, "round_end": int(rr["tick"]),
                    "reason": int(rr["win_reason_code"]), "winner": int(rr["winner_team_code"]),
                    "player_count": int(rr["player_count"])})

with open(os.path.join(ROOT, "logs", "swarm2", "A10_absent_probe.json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1, default=str)
print("\nDONE")
