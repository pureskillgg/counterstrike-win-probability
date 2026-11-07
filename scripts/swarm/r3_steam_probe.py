"""R3 probe 3: is basic_player_personal.steam_id a usable cross-match player key?
Prints only aggregate counts and format checks. No identifier values are printed.
"""
import collections
import os
import re

import pyarrow.parquet as pq

BASE = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
path = os.path.join(BASE, "basic_player_personal.2026-09-18,2026-10-08", "dataframe_00000")
schema = pq.read_schema(path)
print("COLUMNS", [f.name for f in schema])
cols = ["match_key", "round", "player_id", "player_id_fixed", "steam_id", "name"]
table = pq.read_table(path, columns=cols)
print("ROWS", table.num_rows)

mk = table.column("match_key").to_pylist()
sid = table.column("steam_id").to_pylist()
pid = table.column("player_id").to_pylist()
fid = table.column("player_id_fixed").to_pylist()

null_or_blank = sum(1 for s in sid if s is None or str(s).strip() in ("", "0"))
types = collections.Counter(type(s).__name__ for s in sid)
pattern_17 = sum(1 for s in sid if s is not None and re.fullmatch(r"\d{17}", str(s)))
redacted_like = sum(1 for s in sid if s is not None and not re.fullmatch(r"\d+", str(s)))
print("STEAM_ID_TYPES", dict(types))
print("STEAM_ID_NULL_OR_ZERO", null_or_blank)
print("STEAM_ID_17_DIGIT", pattern_17, "NON_NUMERIC_OR_OTHER", redacted_like)

per_match = collections.defaultdict(set)
per_match_pid = collections.defaultdict(lambda: collections.defaultdict(set))
per_match_fid = collections.defaultdict(lambda: collections.defaultdict(set))
for m, s, p, f in zip(mk, sid, pid, fid):
    if s is None:
        continue
    per_match[m].add(s)
    per_match_pid[m][p].add(s)
    per_match_fid[m][f].add(s)

n_matches = len(per_match)
players_per_match = collections.Counter(len(v) for v in per_match.values())
print("MATCHES_WITH_STEAM_ID", n_matches)
print("DISTINCT_STEAM_IDS_PER_MATCH_HIST", sorted(players_per_match.items())[:15])

# Raw slot id -> steam id mapping one-to-one within match?
one_to_one = sum(1 for m in per_match_pid if all(len(v) == 1 for v in per_match_pid[m].values()))
print("MATCHES_RAW_SLOT_TO_ONE_STEAM", one_to_one, "of", len(per_match_pid))
fixed_one = sum(1 for m in per_match_fid if all(len(v) == 1 for v in per_match_fid[m].values()))
print("MATCHES_FIXED_SLOT_TO_ONE_STEAM", fixed_one, "of", len(per_match_fid))

# Cross-match reuse: how many steam ids appear in more than one match?
match_count = collections.Counter()
for m, ids in per_match.items():
    match_count.update(ids)
multi = sum(1 for k, v in match_count.items() if v > 1)
print("STEAM_IDS_TOTAL", len(match_count), "IDS_IN_MORE_THAN_ONE_MATCH", multi)
print("MAX_MATCHES_PER_ID", max(match_count.values()) if match_count else 0)

# Team code / side check on the info table: is team_code the side, and does each slot switch sides?
info = pq.read_table(os.path.join(BASE, "basic_player_info.2026-09-18,2026-10-08", "dataframe_00000"),
                     columns=["match_key", "player_id_fixed", "team_code"])
side_sets = collections.defaultdict(lambda: collections.defaultdict(set))
for m, f, t in zip(info.column("match_key").to_pylist(), info.column("player_id_fixed").to_pylist(),
                   info.column("team_code").to_pylist()):
    side_sets[m][f].add(t)
two_sides = collections.Counter()
for m, d in side_sets.items():
    for f, codes in d.items():
        two_sides[len(codes)] += 1
print("TEAM_CODE_VALUES_PER_SLOT_HIST", dict(two_sides))
codes_all = collections.Counter()
for d in side_sets.values():
    for codes in d.values():
        codes_all.update(codes)
print("TEAM_CODE_VALUES", dict(codes_all))
