import re
import sys
from collections import Counter

import pyarrow.parquet as pq

# Read-only aggregate check. Prints only counts and pattern tallies.
# No individual cell value is printed, stored or echoed.
STEAMID64 = re.compile(r"^7656119\d{10}$")
ANY_DIGITS = re.compile(r"^\d+$")
STEAM2 = re.compile(r"^STEAM_[0-5]:[01]:\d+$")
BOTLIKE = re.compile(r"(?i)^(bot|\(bot\)|bot_)")

paths_file = sys.argv[1]
with open(paths_file, encoding="utf-8") as fh:
    paths = [line.strip() for line in fh if line.strip()]

totals = Counter()
steam_all = []          # kept in memory only, never printed
name_nonempty = 0
player_name_rows_total = 0
player_name_files_nonempty = 0
files_ok = 0
files_missing = 0
schema_sets = Counter()
per_file_map_consistent = 0
per_file_map_checked = 0

for path in paths:
    try:
        tbl = pq.read_table(path)
    except FileNotFoundError:
        files_missing += 1
        continue
    except Exception as exc:
        totals["read_error_" + type(exc).__name__] += 1
        continue
    files_ok += 1
    cols = tuple(tbl.column_names)
    schema_sets[cols] += 1

    if "steam_id" in cols:
        sid = tbl.column("steam_id").to_pylist()
        is_bot = tbl.column("is_bot").to_pylist() if "is_bot" in cols else [False] * len(sid)
        names = tbl.column("name").to_pylist() if "name" in cols else [None] * len(sid)
        pids = tbl.column("player_id").to_pylist() if "player_id" in cols else [None] * len(sid)
        for s, b, n, p in zip(sid, is_bot, names, pids):
            totals["rows"] += 1
            if b:
                totals["is_bot_true"] += 1
            if s is None or s == "":
                totals["steam_id_null_or_empty"] += 1
                continue
            totals["steam_id_nonempty"] += 1
            if STEAMID64.match(s):
                totals["steam_id_matches_SteamID64_pattern"] += 1
            elif ANY_DIGITS.match(s):
                totals["steam_id_all_digits_other_length"] += 1
            elif STEAM2.match(s):
                totals["steam_id_matches_legacy_STEAM_x_pattern"] += 1
            else:
                totals["steam_id_other_format"] += 1
            if b:
                totals["steam_id_nonempty_on_bot_rows"] += 1
            steam_all.append(s)
            if n is not None and str(n).strip() != "":
                name_nonempty += 1
            if n is not None and BOTLIKE.match(str(n)):
                totals["name_bot_like_prefix"] += 1

        # one steam_id per player_id within a match?
        mapping = {}
        ok = True
        for s, p in zip(sid, pids):
            if s is None or p is None:
                continue
            if p in mapping and mapping[p] != s:
                ok = False
                break
            mapping[p] = s
        per_file_map_checked += 1
        if ok:
            per_file_map_consistent += 1

    sibling = path.rsplit("/", 1)[0] + "/player_name"
    try:
        pn = pq.read_table(sibling)
        n_rows = pn.num_rows
        player_name_rows_total += n_rows
        if n_rows > 0:
            player_name_files_nonempty += 1
    except FileNotFoundError:
        totals["player_name_sibling_missing"] += 1
    except Exception as exc:
        totals["player_name_read_error_" + type(exc).__name__] += 1

print("files_read_ok:", files_ok)
print("files_missing:", files_missing)
print("distinct_column_schemas:", len(schema_sets))
for cols, cnt in schema_sets.most_common(3):
    print("  schema x%d: %s" % (cnt, ", ".join(cols)))
for k in sorted(totals):
    print(k + ":", totals[k])
print("name_nonempty_rows:", name_nonempty)
print("distinct_steam_ids_across_checked_files:", len(set(steam_all)))
print("steam_id_rows_for_cross_match_linkage:", len(steam_all))
print("files_with_consistent_player_to_steam_mapping:", per_file_map_consistent, "of", per_file_map_checked)
print("player_name_table_rows_total:", player_name_rows_total)
print("player_name_files_with_rows:", player_name_files_nonempty)
