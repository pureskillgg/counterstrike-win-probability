"""Train-split-only shape counts for steam_id and name in basic_player_personal. Prints counts only."""
import collections
import glob
import json
import os
import re

import pyarrow.parquet as pq

MANIFEST = r"G:\pureskill\csds-cs2\deliverables\email_package_2026-10-19_review\results\split_manifest.json"
TOME = r"G:\pureskill\csds-cs2\tomes\combined\tome\csds"
UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
MAP_RE = re.compile(r"^(de_|cs_)[a-z0-9_]+$", re.I)
TEAM_WORDS = {"ct", "t", "terrorist", "counter-terrorist", "counterterrorist", "team", "bot"}

manifest = json.load(open(MANIFEST, encoding="utf-8"))
train_uuids = set()
for key, label in manifest["assign"].items():
    if label != "train":
        continue
    parts = key.split("/")
    yyyy, mm, dd, uuid = parts[1], parts[2], parts[3], parts[4]
    if yyyy == "2026" and mm == "10" and dd in {"26", "27", "28", "29", "30", "31"}:
        continue
    train_uuids.add(uuid.lower())

steam_len = collections.Counter()
steam_charset = collections.Counter()
name_len_bucket = collections.Counter()
name_has_space = 0
name_map_like = 0
name_team_like = 0
name_bot_prefix = 0
name_all_lower_no_space = 0
distinct_names = set()
distinct_steam = set()
rows = 0
uuid_cache = {}

for page in sorted(glob.glob(os.path.join(TOME, "basic_player_personal.COMBINED", "dataframe_*"))):
    t = pq.read_table(page, columns=["match_key", "steam_id", "name"])
    keys = t.column("match_key").to_pylist()
    steams = t.column("steam_id").to_pylist()
    names = t.column("name").to_pylist()
    for k, s, n in zip(keys, steams, names):
        if k not in uuid_cache:
            m = UUID_RE.search(k) if k is not None else None
            uuid_cache[k] = m.group(0).lower() if m else None
        u = uuid_cache[k]
        if u is None or u not in train_uuids:
            continue
        rows += 1
        s = s or ""
        n = n or ""
        steam_len[len(s)] += 1
        steam_charset["upper" if s.isupper() else ("lower" if s.islower() else "mixed")] += 1
        distinct_steam.add(s)
        distinct_names.add(n)
        L = len(n)
        bucket = "0" if L == 0 else ("1-3" if L <= 3 else ("4-10" if L <= 10 else ("11-20" if L <= 20 else ">20")))
        name_len_bucket[bucket] += 1
        if " " in n:
            name_has_space += 1
        if MAP_RE.match(n):
            name_map_like += 1
        if n.strip().lower() in TEAM_WORDS:
            name_team_like += 1
        if n.lower().startswith("bot"):
            name_bot_prefix += 1
        if n and n == n.lower() and " " not in n:
            name_all_lower_no_space += 1

print(f"train rows={rows}")
print("steam_id length counts:", dict(sorted(steam_len.items())))
print("steam_id case counts:", dict(steam_charset))
print(f"distinct steam_id values={len(distinct_steam)}")
print("name length buckets:", dict(name_len_bucket))
print(f"distinct name values={len(distinct_names)}")
print(f"names containing a space={name_has_space}")
print(f"names matching a map-name pattern={name_map_like}")
print(f"names equal to a team/side word={name_team_like}")
print(f"names starting with 'bot'={name_bot_prefix}")
print(f"names lowercase with no space={name_all_lower_no_space}")
