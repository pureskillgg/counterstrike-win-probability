"""Scan every page of every channel in the given tome collections and list columns whose dtype (or presence) differs.

Footer reads only (no row data). Output: per channel, the columns with more than one dtype / missing in some
collection. Used to design the harmonised combined store (build_combined.py).

  pixi run python -I -X utf8 scripts\\data\\schema_conflicts.py 2024-02-g 2026-10-01_to_08 [more collections]
"""
import collections
import glob
import json
import os
import sys

import pyarrow.parquet as pq

ROOT = r"G:\pureskill\csds-cs2\tomes"
cols = sys.argv[1:] or ["2024-02-g", "2026-10-01_to_08"]


def channels(coll):
    base = os.path.join(ROOT, coll, "tome", "csds")
    out = {}
    for d in sorted(os.listdir(base)):
        p = os.path.join(base, d)
        if not os.path.isdir(p) or d.startswith("."):
            continue
        ch = "header" if d == "header" else d[len("basic_"):].split(".")[0]
        out[ch] = sorted(glob.glob(os.path.join(p, "dataframe_*")))
    return out


per = {c: channels(c) for c in cols}
allch = sorted(set().union(*[set(v) for v in per.values()]))
report = {}
for ch in allch:
    seen = collections.defaultdict(lambda: collections.defaultdict(set))  # column -> collection -> {dtype}
    present = {}
    for c in cols:
        files = per[c].get(ch)
        present[c] = bool(files)
        for f in files or []:
            sch = pq.read_schema(f)
            for name, typ in zip(sch.names, sch.types):
                seen[name][c].add(str(typ))
    conflicts = {}
    for name, d in seen.items():
        alltypes = set().union(*d.values())
        missing_in = [c for c in cols if present[c] and c not in d]
        if len(alltypes) > 1 or missing_in:
            conflicts[name] = {"types": {c: sorted(t) for c, t in d.items()}, "missing_in": missing_in}
    report[ch] = {"present_in": [c for c in cols if present[c]], "n_columns": len(seen), "conflicts": conflicts}

for ch, r in report.items():
    print(f"== {ch}: in {r['present_in']}  cols={r['n_columns']}  conflicts={len(r['conflicts'])}")
    for name, v in r["conflicts"].items():
        print("   ", name, json.dumps(v))
with open(r"G:\pureskill\csds-cs2\data\schema_conflicts.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=1)
