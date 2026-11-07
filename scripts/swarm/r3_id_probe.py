"""R3 probe 2: is there a cross-match player key, and is the in-match slot id
consistent within a match? Reads G: tome pages only, schema plus small aggregates.
"""
import collections
import os

import pyarrow.compute as pc
import pyarrow.parquet as pq

BASE = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"

# 1. Schema inventory: which tables carry any steam / account style key.
keyish = []
for name in sorted(os.listdir(BASE)):
    folder = os.path.join(BASE, name)
    if not os.path.isdir(folder) or name == "header":
        continue
    data = [f for f in os.listdir(folder) if f.startswith("dataframe_")]
    if not data:
        continue
    schema = pq.read_schema(os.path.join(folder, data[0]))
    cols = [f.name for f in schema]
    hits = [c for c in cols if any(k in c.lower() for k in ("steam", "account", "xuid", "uid", "_id", "name"))]
    if hits:
        keyish.append((name.split(".")[0], hits))
print("SCHEMA_ID_COLUMNS")
for name, hits in keyish:
    print(" ", name, hits)

# 2. Within-match consistency of raw slot id vs fixed id, from player_info.
info_path = os.path.join(BASE, "basic_player_info.2026-09-18,2026-10-08", "dataframe_00000")
info = pq.read_table(info_path, columns=["match_key", "round", "player_id", "player_id_fixed", "team_code"])
print("INFO_ROWS", info.num_rows)

by_match = collections.defaultdict(lambda: {"raw": set(), "fixed": set(), "map": collections.defaultdict(set), "teams": collections.defaultdict(set)})
mk = info.column("match_key").to_pylist()
raw = info.column("player_id").to_pylist()
fixed = info.column("player_id_fixed").to_pylist()
team = info.column("team_code").to_pylist()
for m, r, f, t in zip(mk, raw, fixed, team):
    d = by_match[m]
    d["raw"].add(r)
    d["fixed"].add(f)
    d["map"][r].add(f)
    d["teams"][f].add(t)

n = len(by_match)
ten_fixed = sum(1 for d in by_match.values() if len(d["fixed"]) == 10)
raw_fixed_ok = sum(1 for d in by_match.values() if all(len(v) == 1 for v in d["map"].values()))
fixed_team_ok = sum(1 for d in by_match.values() if all(len(v) == 1 for v in d["teams"].values()))
raw_vals = collections.Counter()
for d in by_match.values():
    raw_vals.update(d["raw"])
print("MATCHES", n)
print("MATCHES_WITH_10_FIXED_IDS", ten_fixed)
print("MATCHES_RAW_TO_FIXED_ONE_TO_ONE", raw_fixed_ok)
print("MATCHES_FIXED_ID_ONE_TEAM", fixed_team_ok)
print("RAW_ID_VALUES_ACROSS_MATCHES", sorted(raw_vals.items())[:20])
fixed_vals = collections.Counter()
for d in by_match.values():
    fixed_vals.update(d["fixed"])
print("FIXED_ID_VALUES_ACROSS_MATCHES", sorted(fixed_vals.items())[:20])

# 3. Death table: do raw ids in deaths agree with info mapping in the same match?
death_path = os.path.join(BASE, "basic_player_death.2026-09-18,2026-10-08", "dataframe_00000")
death = pq.read_table(death_path, columns=["match_key", "player_id", "player_id_fixed", "attacker_id", "attacker_id_fixed"])
dm = death.column("match_key").to_pylist()
dr = death.column("player_id").to_pylist()
df_ = death.column("player_id_fixed").to_pylist()
agree = 0
disagree = 0
missing_map = 0
for m, r, f in zip(dm, dr, df_):
    if m not in by_match:
        missing_map += 1
        continue
    mapping = by_match[m]["map"].get(r)
    if mapping is None:
        missing_map += 1
    elif f is not None and f in mapping:
        agree += 1
    else:
        disagree += 1
print("DEATH_ROWS", death.num_rows, "agree", agree, "disagree", disagree, "no_info_mapping", missing_map)
