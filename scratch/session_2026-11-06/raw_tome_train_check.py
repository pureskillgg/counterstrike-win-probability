"""Train-split-only check of raw per-date tomes: counts of distinct name / steam_id values. No values printed."""
import glob
import json
import os
import re

import pyarrow.parquet as pq

MANIFEST = r"G:\pureskill\csds-cs2\deliverables\email_package_2026-10-19_review\results\split_manifest.json"
UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
TOMES = [
    r"G:\pureskill\csds-cs2\tomes\2026-09\tome\csds",
    r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds",
    r"G:\pureskill\csds-cs2\tomes\2024-02\tome\csds",
]

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

for tome in TOMES:
    dirs = glob.glob(os.path.join(tome, "basic_player_personal*"))
    if not dirs:
        print(f"{tome}: no basic_player_personal table")
        continue
    names, steams = set(), set()
    rows = 0
    matched_keys = 0
    seen_keys = 0
    for d in dirs:
        for page in sorted(glob.glob(os.path.join(d, "dataframe_*"))):
            t = pq.read_table(page, columns=["match_key", "steam_id", "name"])
            for k, s, n in zip(t.column("match_key").to_pylist(), t.column("steam_id").to_pylist(), t.column("name").to_pylist()):
                seen_keys += 1
                m = UUID_RE.search(k or "")
                if not m or m.group(0).lower() not in train_uuids:
                    continue
                matched_keys += 1
                rows += 1
                names.add(n or "")
                steams.add(s or "")
    print(f"{os.path.basename(os.path.dirname(os.path.dirname(tome)))}: rows_scanned={seen_keys} train_rows={rows} "
          f"distinct_names={len(names)} distinct_steam_ids={len(steams)}")
