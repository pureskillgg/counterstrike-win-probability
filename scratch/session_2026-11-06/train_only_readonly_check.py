"""Read-only, train-split-only aggregate check on the player tables.

Prints counts and pattern-class counts only. Never prints a match key, player id, steam id or name.
Rows outside the train split (holdout/test, val) and any match dated 2026-10-26..2026-10-31 are excluded.
"""
import collections
import glob
import json
import os
import re

import pyarrow.compute as pc
import pyarrow.parquet as pq

MANIFEST = r"G:\pureskill\csds-cs2\deliverables\email_package_2026-10-19_review\results\split_manifest.json"
TOME = r"G:\pureskill\csds-cs2\tomes\combined\tome\csds"
UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")

# 1. Build the train-only set of match ids from the manifest (ids and dates are not printed).
manifest = json.load(open(MANIFEST, encoding="utf-8"))
train_uuids = set()
excluded_oct = 0
train_total = 0
for key, label in manifest["assign"].items():
    if label != "train":
        continue
    train_total += 1
    parts = key.split("/")
    yyyy, mm, dd, uuid = parts[1], parts[2], parts[3], parts[4]
    if yyyy == "2026" and mm == "10" and dd in {"26", "27", "28", "29", "30", "31"}:
        excluded_oct += 1
        continue
    train_uuids.add(uuid.lower())
print(f"manifest train labels={train_total} train_kept={len(train_uuids)} excluded_oct26_31={excluded_oct}")

# 2. Check how tome match_key values relate to manifest uuids (counts only).
def classify_key(k):
    if k is None:
        return "null"
    if UUID_RE.fullmatch(k):
        return "bare_uuid"
    if UUID_RE.search(k):
        return "contains_uuid"
    return "other"

def steam_class(s):
    if s is None or s == "":
        return "empty"
    if re.fullmatch(r"7656119\d{10}", s):
        return "steamid64_pattern"
    if re.fullmatch(r"STEAM_[0-5]:[01]:\d+", s):
        return "steam2_pattern"
    if re.fullmatch(r"\[U:1:\d+\]", s):
        return "steam3_pattern"
    if re.fullmatch(r"\d+", s):
        return "other_digits"
    if re.fullmatch(r"[A-Za-z]+", s):
        return "letters_only"
    return "mixed_or_other"

personal_pages = sorted(glob.glob(os.path.join(TOME, "basic_player_personal.COMBINED", "dataframe_*")))
uuid_cache = {}
key_class = collections.Counter()
steam_class_counts = collections.Counter()
name_empty = 0
name_nonempty = 0
train_rows = 0
steam_ids_train = set()
match_player_pairs = set()
steam_to_matches = collections.defaultdict(set)

for page in personal_pages:
    t = pq.read_table(page, columns=["match_key", "player_id", "steam_id", "name"])
    keys = t.column("match_key").to_pylist()
    for k in set(keys):
        key_class[classify_key(k)] += 1
    # Filter in Python on the normalised uuid so the match key text is never printed.
    pids = t.column("player_id").to_pylist()
    steams = t.column("steam_id").to_pylist()
    names = t.column("name").to_pylist()
    for k, pid, s, n in zip(keys, pids, steams, names):
        uuid_part = uuid_cache.get(k)
        if uuid_part is None and k not in uuid_cache:
            m = UUID_RE.search(k) if k is not None else None
            uuid_part = m.group(0).lower() if m else None
            uuid_cache[k] = uuid_part
        if uuid_part is None or uuid_part not in train_uuids:
            continue
        train_rows += 1
        steam_class_counts[steam_class(s)] += 1
        if s:
            steam_ids_train.add(s)
            steam_to_matches[s].add(uuid_part)
        match_player_pairs.add((uuid_part, pid))
        if n:
            name_nonempty += 1
        else:
            name_empty += 1

print("personal table: match_key shape counts over distinct keys:", dict(key_class))
print(f"personal table train-split rows={train_rows}")
print("steam_id pattern classes (train rows):", dict(steam_class_counts))
print(f"distinct steam_id values (train)={len(steam_ids_train)}")
print(f"distinct (match, player_id) pairs (train)={len(match_player_pairs)}")
multi = sum(1 for s, ms in steam_to_matches.items() if len(ms) > 1)
print(f"steam_id values seen in more than one train match={multi}")
print(f"name non-empty rows (train)={name_nonempty} empty={name_empty}")

# 3. basic_player_info: distinct id counts on train rows only (schema already confirmed).
info_pages = sorted(glob.glob(os.path.join(TOME, "basic_player_info.COMBINED", "dataframe_*")))
info_rows = 0
info_pids = set()
info_ctrl = set()
info_rank_types = set()
for page in info_pages:
    t = pq.read_table(page, columns=["match_key", "player_id", "player_controller_id", "rank_type"])
    keys = t.column("match_key").to_pylist()
    pids = t.column("player_id").to_pylist()
    ctrls = t.column("player_controller_id").to_pylist()
    rts = t.column("rank_type").to_pylist()
    for k, pid, c, rt in zip(keys, pids, ctrls, rts):
        m = UUID_RE.search(k or "")
        if not m or m.group(0).lower() not in train_uuids:
            continue
        info_rows += 1
        info_pids.add(pid)
        info_ctrl.add(c)
        info_rank_types.add(rt)
print(f"basic_player_info train-split rows={info_rows}")
print(f"basic_player_info distinct player_id (train)={len(info_pids)}")
print(f"basic_player_info distinct player_controller_id (train)={len(info_ctrl)}")
print(f"basic_player_info distinct rank_type codes (train)={len(info_rank_types)}")
