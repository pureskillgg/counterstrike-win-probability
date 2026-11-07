"""Count-only linkage check on one raw training-window file.

Prints counts and pattern-match rates. Never prints a key, name, Steam ID or
any other cell value. Frozen-test keys are excluded before any counting.
"""
import json
import re
import subprocess
import sys

import pyarrow.parquet as pq
import pyarrow.compute as pc

sys.stdout.reconfigure(encoding="utf-8")

RAW = r"G:/pureskill/csds-cs2/evac_2026-10-13/tomes/2026-10-01_to_08/tome/csds/basic_player_personal.2026-09-18,2026-10-08/dataframe_00000"
FROZEN = r"C:/Users/<user>/Documents/pureskill/counterstrike-win-probability/data/benchmark/frozen_test_keys.json"
AWS = ["aws", "--profile", "runpod", "--endpoint-url", "https://s3api-eu-ro-1.runpod.io", "--region", "eu-ro-1"]

MATCH_RE = re.compile(r"^csds/\d{4}/\d{2}/\d{2}/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/csds$")
STEAM64_RE = re.compile(r"^7656119\d{10}$")

# frozen-test keys (never printed)
fz = json.load(open(FROZEN, encoding="utf-8"))
if isinstance(fz, dict):
    frozen = set()
    for v in fz.values():
        if isinstance(v, list):
            frozen.update(x for x in v if isinstance(x, str))
    if not frozen:
        frozen = set(fz.keys())
else:
    frozen = set(x for x in fz if isinstance(x, str))
print("frozen-test keys loaded:", len(frozen))

t = pq.read_table(RAW, columns=["match_key", "steam_id", "name", "is_bot"])
print("raw personal rows in file:", t.num_rows)
mk = t.column("match_key").to_pylist()
sid = t.column("steam_id").to_pylist()
nm = t.column("name").to_pylist()
bot = t.column("is_bot").to_pylist()

keep = [i for i, k in enumerate(mk) if k not in frozen and k is not None and not str(k) >= "csds/2026/10/26"]
print("rows after excluding frozen-test and Oct 26+ keys:", len(keep), "(dropped:", t.num_rows - len(keep), ")")
kept_keys = set(mk[i] for i in keep)
print("distinct match keys in kept rows:", len(kept_keys))
print("match-key format ok (share of kept rows):", round(sum(bool(MATCH_RE.match(mk[i])) for i in keep) / max(len(keep), 1), 4))
print("steam_id non-empty rows:", sum(1 for i in keep if sid[i]))
s64 = [sid[i] for i in keep if sid[i] and STEAM64_RE.match(str(sid[i]))]
print("steam_id rows matching SteamID64 pattern (7656119 + 10 digits):", len(s64), "of", sum(1 for i in keep if sid[i]))
print("distinct steam_id values:", len(set(sid[i] for i in keep if sid[i])))
print("name non-empty rows:", sum(1 for i in keep if nm[i]))
print("is_bot true rows:", sum(1 for i in keep if bot[i]))

# export match keys: count overlap only
raw = subprocess.run(AWS + ["s3", "cp", "s3://z7yvc40ykx/export_clean_v5/match_keys.json", "-"], capture_output=True, check=True).stdout
exp = json.loads(raw)
exp_set = set(x for x in exp if isinstance(x, str))
print("export_clean_v5 match keys:", len(exp_set), "format ok share:", round(sum(bool(MATCH_RE.match(x)) for x in exp_set) / max(len(exp_set), 1), 4))
print("export match keys present in this raw file (non-frozen):", len(exp_set & kept_keys))
print("raw kept match keys present in export key list:", len(kept_keys & exp_set), "of", len(kept_keys))
