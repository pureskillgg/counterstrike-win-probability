import json
import re
import sys
import glob
import collections
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8")

SCR = r"C:\Users\13144\AppData\Local\Temp\claude\44708acea14da7d4\047ee5fa-b35b-4b25-97b1-beaf96834ff2\scratchpad"
assign = json.load(open(SCR + r"\split_manifest.json", encoding="utf-8"))["assign"]
ROOT = r"G:\pureskill\csds-cs2\tomes"

def in_window(d):
    return d is not None and "2026-10-26" <= d[:10] <= "2026-10-31"

valve_re = re.compile(r"^Valve Counter-Strike 2 [a-z0-9_\-]+ Server \(srcds\d+[^)]*\)$")
faceit_re = re.compile(r"^FACEIT\.com")

# 1. the 14 non-template matches (train/val, outside window), with their batch table path
targets = {}
for p in sorted(glob.glob(ROOT + r"\*\tome\csds\header\dataframe_*")):
    t = pq.read_table(p, columns=["key", "match_date", "server_name"])
    for k, d, s in zip(t.column("key").to_pylist(), t.column("match_date").to_pylist(),
                       t.column("server_name").to_pylist()):
        if assign.get(k) not in ("train", "val") or in_window(d):
            continue
        if valve_re.match(s) or faceit_re.match(s):
            continue
        targets[k] = (s, p)
print("target matches:", len(targets))

# 2. player name tables in the same batch: schema only first
batches = sorted(set(v[1].split(r"\tome\csds")[0] for v in targets.values()))
print("batches containing targets:", len(batches), [b.split("\\")[-1] for b in batches])

def player_name_files(batch_dir):
    return sorted(glob.glob(batch_dir + r"\tome\csds\basic_player_name.*\dataframe_*"))

for b in batches:
    fs = player_name_files(b)
    if not fs:
        print("no basic_player_name table in", b.split("\\")[-1])
        continue
    sch = pq.ParquetFile(fs[0]).schema_arrow
    print(b.split("\\")[-1], "player_name columns:", sch.names)
