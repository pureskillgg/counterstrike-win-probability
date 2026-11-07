import json, os, random, re, collections
import numpy as np
import pyarrow.parquet as pq

BASE = "C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/match_key_audit"
RAW = "E:/csds_final"
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-([0-9a-f])[0-9a-f]{3}-([0-9a-f])[0-9a-f]{3}-[0-9a-f]{12}$")
KEY_RE = re.compile(r"^csds/(\d{4})/(\d{2})/(\d{2})/([0-9a-f-]{36})/csds$")

keys = json.load(open(os.path.join(BASE, "export_clean_v5", "match_keys.json"), encoding="utf-8"))
split = np.load(os.path.join(BASE, "export_clean_v5", "match_split.npy"))
frozen = set(json.load(open(os.path.join(BASE, "cs2data", "keys", "frozen_test_keys.json"), encoding="utf-8")))
assert len(keys) == len(split)

# 1. Format and UUID version/variant census over all export keys (no values printed)
fmt_ok = 0
ver = collections.Counter()
var = collections.Counter()
for k in keys:
    m = KEY_RE.match(k)
    if m:
        fmt_ok += 1
        u = UUID_RE.match(m.group(4))
        if u:
            ver[u.group(1)] += 1
            var[u.group(2)] += 1
print("export keys:", len(keys), "format-match:", fmt_ok)
print("UUID version nibble census:", dict(ver))
print("UUID variant nibble census:", dict(var))
print("unique keys:", len(set(keys)))

# 2. Frozen-set census (count only, no values)
print("frozen keys in frozen_test_keys.json:", len(frozen), "frozen format-match:",
      sum(1 for k in frozen if KEY_RE.match(k)))
print("export keys that are in frozen set:", sum(1 for k in keys if k in frozen))

# 3. Split code census vs frozen (counts only)
print("split codes:", collections.Counter(split.tolist()))

# 4. Eligible sample: September 2026, not test-split code 2, not frozen, not Oct
elig = []
for i, k in enumerate(keys):
    m = KEY_RE.match(k)
    if not m:
        continue
    y, mo, d = m.group(1), m.group(2), m.group(3)
    if y == "2026" and mo == "09" and int(split[i]) in (0, 1) and k not in frozen:
        elig.append((i, k))
print("eligible Sept-2026 train/val non-frozen keys:", len(elig))
rng = random.Random(12345)
sample = rng.sample(elig, 20)

res = collections.Counter()
personal_schema = {}
for idx, (i, k) in enumerate(sample):
    m = KEY_RE.match(k)
    path = os.path.join(RAW, "csds", m.group(1), m.group(2), m.group(3), m.group(4))
    exists = os.path.isdir(path)
    if not exists:
        res["folder_missing"] += 1
        print(f"[{idx:02d}] split={int(split[i])} folder_exists=False")
        continue
    files = set(os.listdir(path))
    has_status = "player_status" in files
    tables = {t: (t in files) for t in ["player_personal", "player_name", "player_chat", "header", "player_status"]}
    res["folder_exists"] += 1
    res["has_player_status"] += int(has_status)
    res["entries_" + str(len(files))] += 1
    print(f"[{idx:02d}] split={int(split[i])} folder_exists=True entries={len(files)} tables={tables}")
    if idx < 3:
        for t in ["player_personal", "player_name", "player_chat", "header", "player_status"]:
            fp = os.path.join(path, t)
            if os.path.isfile(fp):
                sch = pq.read_schema(fp)
                md = pq.read_metadata(fp)
                personal_schema.setdefault(t, []).append(
                    (md.num_rows, [(f.name, str(f.type)) for f in sch])
                )
print("sample summary:", dict(res))
for t, lst in personal_schema.items():
    print("SCHEMA", t, "rows(metadata of first sampled folder)=", lst[0][0], "cols=", lst[0][1])
