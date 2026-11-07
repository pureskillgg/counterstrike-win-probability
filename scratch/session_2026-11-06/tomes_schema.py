"""Schema-only inventory of raw tome tables (footers only, no row values).

For every top-level tome folder and every csds/<table> directory, record file
count, bytes and the column signature of the first dataframe_* parquet found.
Folder eligibility for later value scans is decided here by name.
"""
import os
import re
import sys
import json
import collections
import pyarrow.parquet as pq

T = "C:/Users/13144/Documents/pureskill_internship/csds-cs2/tomes"
OUT = sys.argv[1]
COL_SCREEN = re.compile(
    r"steam|steamid|sid|accountid|account|name|nick|player|user|username|ip|email|phone|"
    r"address|chat|message|path|host|demo|file|url|avatar", re.I)
OCT22_31 = re.compile(r"2026-10-(2[2-9]|3[01])")

tables = collections.OrderedDict()
folders = sorted(os.listdir(T))
for folder in folders:
    fpath = os.path.join(T, folder)
    if not os.path.isdir(fpath):
        continue
    eligible = not OCT22_31.search(folder)
    # Folders that cover Oct 22-31 are schema-only regardless of contents.
    for dp, dns, fns in os.walk(fpath):
        rel = os.path.relpath(dp, fpath).replace("\\", "/")
        parts = rel.split("/")
        # Only <...>/csds/<table> directories. Nested header/ dirs under a table are skipped.
        if "csds" not in parts:
            continue
        i = parts.index("csds")
        if len(parts) != i + 2:
            continue
        tname = re.sub(r"\.[0-9]{4}-[0-9]{2}-[0-9]{2}.*$", "", parts[i + 1])
        dfs = sorted(f for f in fns if f.startswith("dataframe_") and not f.endswith(".json"))
        if not dfs:
            continue
        key = tname
        rec = tables.setdefault(key, {"folders": set(), "files": 0, "bytes": 0,
                                      "signatures": collections.Counter(), "eligible_files": 0,
                                      "schema_example": None, "tome_json_keys": set()})
        rec["folders"].add(folder)
        for f in dfs:
            fp = os.path.join(dp, f)
            rec["files"] += 1
            rec["bytes"] += os.path.getsize(fp)
            if eligible:
                rec["eligible_files"] += 1
        # one footer per (table type, folder) to capture schema variants
        try:
            pf = pq.ParquetFile(os.path.join(dp, dfs[0]))
            sig = tuple((f.name, str(f.type)) for f in pf.schema_arrow)
            rec["signatures"][sig] += 1
            if rec["schema_example"] is None:
                rec["schema_example"] = {"cols": [c for c, _ in sig], "types": {c: t for c, t in sig}, "rows_first_file": pf.metadata.num_rows, "folder": folder}
        except Exception as e:  # noqa
            rec["signatures"][("ERR", type(e).__name__)] += 1

out = {}
for k, v in tables.items():
    sig_list = [dict(cols=[c for c, _ in s], types={c: t for c, t in s}, n=n) for s, n in v["signatures"].items()]
    out[k] = {"n_folders": len(v["folders"]), "files": v["files"], "eligible_files": v["eligible_files"],
              "bytes": v["bytes"], "n_signatures": len(sig_list),
              "col_screen_hits": [c for c in (v["schema_example"] or {}).get("cols", []) if COL_SCREEN.search(c)],
              "cols": (v["schema_example"] or {}).get("cols", []),
              "types": (v["schema_example"] or {}).get("types", {}),
              "example_folder": (v["schema_example"] or {}).get("folder"),
              "signatures": sig_list}
json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, default=str)
print("table types:", len(out))
for k in sorted(out):
    v = out[k]
    print(f"{k:34s} folders={v['n_folders']:3d} files={v['files']:6d} elig_files={v['eligible_files']:6d} MB={v['bytes']/1e6:9.1f} sigs={v['n_signatures']} screen={v['col_screen_hits']}")
