import json, os, re, collections
import numpy as np
import pyarrow.parquet as pq
BASE = "C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/match_key_audit"
RAW = "E:/csds_final/csds"
KEY_RE = re.compile(r"^csds/(\d{4})/(\d{2})/(\d{2})/([0-9a-f-]{36})/csds$")
keys = json.load(open(os.path.join(BASE, "export_clean_v5", "match_keys.json"), encoding="utf-8"))
split = np.load(os.path.join(BASE, "export_clean_v5", "match_split.npy"))
frozen = set(json.load(open(os.path.join(BASE, "cs2data", "keys", "frozen_test_keys.json"), encoding="utf-8")))
elig = [k for i, k in enumerate(keys) if (m := KEY_RE.match(k)) and m.group(1) == "2026" and m.group(2) == "09" and int(split[i]) in (0, 1) and k not in frozen]
exists = 0; has_ps = 0; has_pp = 0; steam_col = 0; name_col = 0; pp_rows = 0
for k in elig:
    m = KEY_RE.match(k)
    p = os.path.join(RAW, m.group(1), m.group(2), m.group(3), m.group(4))
    if not os.path.isdir(p): continue
    exists += 1
    files = set(os.listdir(p))
    has_ps += "player_status" in files
    if "player_personal" in files:
        has_pp += 1
        names = pq.read_schema(os.path.join(p, "player_personal")).names
        steam_col += "steam_id" in names
        name_col += "name" in names
        pp_rows += pq.read_metadata(os.path.join(p, "player_personal")).num_rows
print("eligible Sept train/val non-frozen:", len(elig))
print("folder exists:", exists, "| player_status present:", has_ps, "| player_personal present:", has_pp)
print("player_personal schema has steam_id column:", steam_col, "| has name column:", name_col)
print("total player_personal rows (metadata only) across eligible folders:", pp_rows)
# Which split codes are the eligible keys (counts only)
print("split codes of eligible:", collections.Counter(int(split[i]) for i,k in enumerate(keys) if k in set(elig)))
