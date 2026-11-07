"""Schema-only pass over git-tracked data files in the repo (step 0 + step 1).
Reads CSV headers and JSON structure only. No row values are read or printed.
"""
import json
import os
import re
import subprocess
import sys

REPO = "C:/Users/13144/Documents/pureskill/counterstrike-win-probability"
COL_SCREEN = re.compile(
    r"steam|steamid|sid|accountid|account|name|nick|player|user|username|ip|email|phone|"
    r"address|chat|message|path|host|demo|file|url|avatar", re.I)
EXT = re.compile(r"\.(parquet|csv|json|jsonl|dem|txt|log)$", re.I)

files = subprocess.run(["git", "-C", REPO, "ls-files"], capture_output=True, text=True).stdout.split("\n")
files = [f for f in files if f and EXT.search(f)]
print("tracked data files:", len(files))
for f in files:
    p = os.path.join(REPO, f)
    size = os.path.getsize(p) if os.path.exists(p) else -1
    kind = f.rsplit(".", 1)[-1].lower()
    line = f"{f}  size={size}"
    if kind == "csv":
        with open(p, "rb") as fh:
            header = fh.readline().decode("utf-8", "replace").rstrip("\r\n")
            n = sum(1 for _ in fh)
        cols = [c.strip().strip('"') for c in header.split(",")]
        hits = [c for c in cols if COL_SCREEN.search(c)]
        line += f"  csv rows={n} ncols={len(cols)} screen={hits} cols={cols[:14]}{'...' if len(cols)>14 else ''}"
    elif kind == "json":
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception as e:  # noqa
            line += f"  json parse error {type(e).__name__}"
            print(line)
            continue
        if isinstance(d, dict):
            hits = [k for k in d.keys() if COL_SCREEN.search(str(k))]
            line += f"  json dict keys={len(d)} screen={hits} keys={list(d.keys())[:16]}"
        elif isinstance(d, list):
            first = d[0] if d else None
            if isinstance(first, dict):
                hits = [k for k in first.keys() if COL_SCREEN.search(str(k))]
                line += f"  json list len={len(d)} elem_keys={list(first.keys())[:16]} screen={hits}"
            else:
                line += f"  json list len={len(d)} elem_type={type(first).__name__}"
        else:
            line += f"  json type={type(d).__name__}"
    elif kind in ("txt", "log"):
        with open(p, "rb") as fh:
            n = sum(1 for _ in fh)
        line += f"  {kind} lines={n} (not value-scanned)"
    else:
        line += f"  {kind}"
    print(line)
