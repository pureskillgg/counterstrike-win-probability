"""Read-only PII audit, step 1-2: file inventory and column-name screen.

Footer/schema-only. Reads no row values. Writes JSON to the scratchpad.
Run with: python.exe -I pii_inventory.py
"""
import os
import re
import json
import sys
import collections
import pyarrow.parquet as pq

ROOTS = [
    "C:/Users/13144/Documents/pureskill_internship/csds-cs2/data",  # target of G:/pureskill/csds-cs2/data symlink
    "G:/pureskill/csds-cs2/scripts",
    "G:/pureskill/csds-cs2/models",
    "G:/pureskill/csds-cs2/notes",
    "G:/pureskill/csds-cs2/logs",
    "G:/pureskill/csds-cs2/deliverables",
    "G:/pureskill/csds-cs2/merged_2024",
    "G:/pureskill/csds-cs2/source_2024",
    "G:/pureskill/csds-cs2/staging_cs2data_gz",
]
ROOT_FILES = [
    "G:/pureskill/csds-cs2/scratchpad_val_failure_join.parquet",
    "G:/pureskill/csds-cs2/HANDOFF.md",
]
DATA_EXT = {".parquet", ".csv", ".json", ".jsonl", ".dem", ".txt", ".log", ".tsv",
            ".npz", ".npy", ".gz", ".pkl", ".feather", ".md", ".h5"}
SKIP_DIRS = {".git", ".pixi", "envs", ".meridian-outputs-cache", "__pycache__", "catboost_info"}

COL_SCREEN = re.compile(
    r"steam|steamid|sid|accountid|account|name|nick|player|user|username|ip|email|phone|"
    r"address|chat|message|path|host|demo|file|url|avatar", re.I)
META_PAT = re.compile(r"C:\\\\Users|C:\\Users|/home/|/Users/|@[A-Za-z0-9.-]+\.[a-z]{2,}|steamcommunity|7656119\d{10}", re.I)


def walk():
    seen = set()
    for r in ROOTS:
        if not os.path.isdir(r):
            print("MISSING ROOT", r)
            continue
        for dp, dns, fns in os.walk(r):
            dns[:] = [d for d in dns if d not in SKIP_DIRS]
            for fn in fns:
                ext = os.path.splitext(fn)[1].lower()
                if ext in DATA_EXT:
                    p = os.path.join(dp, fn).replace("\\", "/")
                    if p not in seen:
                        seen.add(p)
                        yield p
    for f in ROOT_FILES:
        if os.path.isfile(f):
            yield f


def parquet_info(p):
    pf = pq.ParquetFile(p)
    md = pf.metadata
    sch = pf.schema_arrow
    cols = [(f.name, str(f.type)) for f in sch]
    kv = {}
    meta = sch.metadata or {}
    for k, v in meta.items():
        kk = k.decode("utf-8", "replace")
        vv = v.decode("utf-8", "replace")
        kv[kk] = {"len": len(vv), "meta_pat_hits": len(META_PAT.findall(vv))}
    return {
        "kind": "parquet",
        "rows": md.num_rows,
        "row_groups": md.num_row_groups,
        "ncols": len(cols),
        "columns": cols,
        "col_screen_hits": [c for c, _ in cols if COL_SCREEN.search(c)],
        "kv_meta": kv,
    }


def json_info(p, size):
    if size > 200 * 1024 * 1024:
        return {"kind": "json", "note": "too large to parse; schema not extracted"}
    try:
        with open(p, "r", encoding="utf-8") as fh:
            d = json.load(fh)
    except Exception as e:  # noqa
        return {"kind": "json", "note": f"parse error: {type(e).__name__}"}
    if isinstance(d, dict):
        return {"kind": "json", "top": "dict", "n_keys": len(d),
                "keys": list(d.keys())[:40],
                "col_screen_hits": [k for k in d.keys() if COL_SCREEN.search(str(k))][:20]}
    if isinstance(d, list):
        first = type(d[0]).__name__ if d else None
        info = {"kind": "json", "top": "list", "len": len(d), "first_elem_type": first}
        if d and isinstance(d[0], dict):
            info["elem_keys"] = list(d[0].keys())[:40]
            info["col_screen_hits"] = [k for k in d[0].keys() if COL_SCREEN.search(str(k))]
        return info
    return {"kind": "json", "top": type(d).__name__}


def csv_info(p, size):
    if size > 1024 * 1024 * 1024:
        return {"kind": "csv", "note": "too large; header only"}
    with open(p, "rb") as fh:
        header = fh.readline().decode("utf-8", "replace").rstrip("\r\n")
        n = 0
        for _ in fh:
            n += 1
    cols = [c.strip().strip('"') for c in header.split(",")]
    return {"kind": "csv", "rows": n, "ncols": len(cols), "columns": cols,
            "col_screen_hits": [c for c in cols if COL_SCREEN.search(c)]}


def generic_info(p, size):
    return {"kind": os.path.splitext(p)[1].lower().lstrip("."), "size": size}


def main(out_path):
    out = []
    for p in walk():
        try:
            size = os.path.getsize(p)
        except OSError:
            size = -1
        ext = os.path.splitext(p)[1].lower()
        rec = {"path": p, "size": size}
        try:
            if ext == ".parquet":
                rec.update(parquet_info(p))
            elif ext == ".json":
                rec.update(json_info(p, size))
            elif ext in {".csv", ".tsv"}:
                rec.update(csv_info(p, size))
            else:
                rec.update(generic_info(p, size))
        except Exception as e:  # noqa
            rec["error"] = f"{type(e).__name__}: {str(e)[:160]}"
        out.append(rec)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=str)
    kinds = collections.Counter(r.get("kind", "?") for r in out)
    print("files:", len(out), dict(kinds))
    print("json written:", out_path)


if __name__ == "__main__":
    main(sys.argv[1])
