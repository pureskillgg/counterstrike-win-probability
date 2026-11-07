"""A10 round-16 probe (read-only, raw E: channels for one match).

Question: which player is missing from player_status at the live start of round 16 of match
m_7392c5c6d537a57c2e735e98, and is that missing row explained by a death, a team
change, a disconnect, or a status gap? Reads only the channels needed, column-projected.
"""
import collections
import os
import sys

import polars as pl
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8")

MATCH_DIR = r"E:\csds_final\csds\2026\10\07\m_7392c5c6d537a57c2e735e98"
MK = "csds/2026/10/07/m_7392c5c6d537a57c2e735e98/csds"
TICK_STEP = 64
START = 5
T, CT = 2, 3
FOCUS = [14, 15, 16, 17]


def rd(ch, cols=None):
    path = os.path.join(MATCH_DIR, ch)
    tbl = pq.read_table(path, columns=cols)
    return pl.from_arrow(tbl)


print("== events around the boundary ==")
rs = rd("round_state", ["round", "tick", "event_type", "t_score", "ct_score", "phase", "is_warmup"])
rst = rd("round_start", ["round", "tick"])
re_ = rd("round_end", ["round", "tick", "winner_team_code", "win_reason_code", "player_count"])
for r in FOCUS:
    print(f"round {r}:")
    print("  round_state:", rs.filter(pl.col("round") == r).sort("tick").select(
        ["tick", "event_type", "t_score", "ct_score", "phase", "is_warmup"]).rows())
    print("  round_start:", rst.filter(pl.col("round") == r).rows())
    print("  round_end:", re_.filter(pl.col("round") == r).rows())

print("\n== player_info (roster) for rounds 14-17 ==")
pi = rd("player_info", ["round", "player_id", "player_id_fixed", "team_code"])
print(pi.filter(pl.col("round").is_in(FOCUS)).sort(["round", "team_code", "player_id"]).rows())

print("\n== player_death rounds 13-17 ==")
de = rd("player_death", ["round", "tick", "player_id", "player_id_fixed", "attacker_id",
                         "player_team_code", "weapon_name", "second"])
print(de.filter(pl.col("round").is_between(13, 17)).sort(["round", "tick"]).rows())

print("\n== team_change, player_disconnect, bomb_state (whole match) ==")
tc = rd("team_change", ["round", "tick", "player_id", "player_id_fixed", "old_team_code", "new_team_code",
                        "is_bot", "silent"])
print("team_change:", tc.sort(["round", "tick"]).rows())
dc = rd("player_disconnect", ["round", "tick", "player_id", "player_id_fixed", "disconnect_reason",
                              "is_bot"])
print("player_disconnect:", dc.sort(["round", "tick"]).rows())
bs = rd("bomb_state", ["round", "tick", "event_type", "player_id", "site_code"])
print("bomb_state rounds 13-17:", bs.filter(pl.col("round").is_between(13, 17)).sort(["round", "tick"]).rows())

print("\n== player_status / player_vector for round 16 ==")
st = rd("player_status", ["round", "tick", "player_id", "player_id_fixed", "health", "armor",
                          "money", "equipment_value_calc"]).filter(pl.col("round") == 16)
vec = rd("player_vector", ["round", "tick", "player_id", "team_code"]).filter(pl.col("round") == 16)
fz_row = rs.filter((pl.col("round") == 16) & (pl.col("event_type") == "round_freeze_end")).sort("tick")
fz = int(fz_row["tick"][0])
end = int(re_.filter(pl.col("round") == 16)["tick"][0])
print(f"round 16 live start fz={fz} end={end} live_ticks={end - fz} (~{(end - fz) / 64:.2f} s)")
print(f"round 16 freeze_end rows: {fz_row.height}")
st_tick_min, st_tick_max = int(st["tick"].min()), int(st["tick"].max())
print(f"player_status round 16 tick range: {st_tick_min}..{st_tick_max}; rows {st.height}")
print(f"  rows with tick < fz: {st.filter(pl.col('tick') < fz).height}; "
      f"rows with tick >= end: {st.filter(pl.col('tick') >= end).height}")

roster16 = pi.filter(pl.col("round") == 16)
print("roster round 16 (player_info):", roster16.sort(["team_code", "player_id"]).rows())

# status presence per player at snapshot ticks
snap_ticks = list(range(fz, end, TICK_STEP))
print(f"snapshot ticks in round 16: {len(snap_ticks)}")
first_ticks = snap_ticks[:3]
for t in first_ticks:
    pres = st.filter(pl.col("tick") == t).select(["player_id", "player_id_fixed", "health"]).sort("player_id")
    print(f"  tick {t} (k={(t - fz) // TICK_STEP}): status rows {pres.height}; ids "
          f"{pres['player_id'].to_list()}")
    v = vec.filter(pl.col("tick") == t).select(["player_id", "team_code"]).sort("player_id")
    print(f"    vector rows {v.height}; team counts {v['team_code'].value_counts().sort('team_code').rows()}")

# per-player presence over the round
pres_all = st.group_by("player_id").agg([
    pl.len().alias("rows"), pl.col("tick").min().alias("tmin"), pl.col("tick").max().alias("tmax"),
    pl.col("player_id_fixed").n_unique().alias("n_fixed"),
    (pl.col("tick") == fz).sum().alias("rows_at_fz")]).sort("player_id")
print("\nper-player status presence in round 16:")
print(pres_all.rows())

# which roster players are absent at the first snapshot
at_fz_ids = set(st.filter(pl.col("tick") == fz)["player_id"].to_list())
roster_ids = roster16["player_id"].to_list()
absent = [(pid, team) for pid, team in roster16.select(["player_id", "team_code"]).rows()
          if pid not in at_fz_ids]
print(f"\nroster ids absent from player_status at fz: {absent}")
print(f"roster ids present at fz: {len(at_fz_ids & set(roster_ids))}; roster size {len(roster_ids)}")

# per-tick presence for absent ids over the first 10 snapshots
for pid, team in absent:
    rows = st.filter(pl.col("player_id") == pid).sort("tick")
    print(f"\nabsent player {pid} (team {team}): status rows in round 16 = {rows.height}")
    if rows.height:
        print("  first rows:", rows.select(["tick", "health", "armor", "money"]).head(5).rows())
        print("  last rows:", rows.select(["tick", "health", "armor", "money"]).tail(3).rows())
    print(f"  any death event in round 16 for this id: "
          f"{de.filter((pl.col('round') == 16) & ((pl.col('player_id') == pid) | (pl.col('player_id_fixed') == pid))).rows()}")
    print(f"  death events (any round) for this id in rounds 14-17: "
          f"{de.filter(pl.col('round').is_between(14, 17) & ((pl.col('player_id') == pid) | (pl.col('player_id_fixed') == pid))).rows()}")
    print(f"  team_change rows for this id: {tc.filter((pl.col('player_id') == pid) | (pl.col('player_id_fixed') == pid)).rows()}")
    print(f"  player_info rows for this id (all rounds): "
          f"{pi.filter((pl.col('player_id') == pid) | (pl.col('player_id_fixed') == pid)).select(['round','team_code']).rows()}")
    # which status ids appear at the same time that are not in the roster
    extra = set(at_fz_ids) - set(roster_ids)
    print(f"  status ids at fz not in roster (round 16 player_info): {sorted(extra)}")

print("\nDONE")
