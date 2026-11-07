import json
import re
import collections
import sys

SCR = "C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad"
path = sys.argv[1] if len(sys.argv) > 1 else SCR + "/value_scan.jsonl"
rows = [json.loads(l) for l in open(path, encoding="utf-8")]


def group(p):
    p = p.replace("\\", "/")
    if "csds-cs2/data/" in p:
        rest = p.split("csds-cs2/data/")[1]
        return "data/" + rest.split("/")[0]
    if "csds-cs2/" in p:
        return "root/" + p.split("csds-cs2/")[1].split("/")[0]
    return p[:40]


agg = collections.OrderedDict()
for d in rows:
    if "error" in d:
        continue
    g = group(d["path"])
    a = agg.setdefault(g, {"files": 0, "rows_total": 0, "excl_fh": 0, "excl_oct": 0, "scanned": 0,
                           "key_nonuuid": 0, "str": collections.defaultdict(collections.Counter),
                           "str_distinct_max": collections.defaultdict(int),
                           "id": collections.defaultdict(collections.Counter)})
    a["files"] += 1
    a["rows_total"] += d["rows_total"]
    a["excl_fh"] += d["n_frozen_holdout_rows_excluded"]
    a["excl_oct"] += d["n_oct26_31_rows_excluded"]
    a["scanned"] += d["rows_scanned"]
    a["key_nonuuid"] += d["key_nonuuid_shape_count"]
    for c, v in d["str_cols"].items():
        for pn, n in v["pattern_rows"].items():
            a["str"][c][pn] += n
        a["str"][c]["_rows"] += v["rows"]
        a["str_distinct_max"][c] = max(a["str_distinct_max"][c], v["n_distinct"])
    for c, v in d["num_cols"].items():
        if c in ("round", "tick", "second", "rid", "tick_rate"):
            pass
        a["id"][c]["non_null"] += v["non_null"]
        a["id"][c]["steam_range"] += v["steamid64_range_rows"]
        a["id"][c]["ge_1e7"] += v["ge_1e7_rows"]
        a["id"][c]["distinct_max"] = max(a["id"][c].get("distinct_max", 0), v["distinct_count"] or 0)

for g, a in agg.items():
    print(f"{g:34s} files={a['files']:4d} rows_total={a['rows_total']:>11d} excl_fh={a['excl_fh']:>10d} excl_oct={a['excl_oct']:>9d} scanned={a['scanned']:>11d} key_nonuuid={a['key_nonuuid']}")
    for c, cnt in a["str"].items():
        pats = {k: v for k, v in cnt.items() if k != "_rows" and v}
        print(f"     str {c:14s} rows={cnt['_rows']} distinct_max_per_file={a['str_distinct_max'][c]} pattern_rows={pats}")
    for c, cnt in a["id"].items():
        if cnt["steam_range"] or cnt["ge_1e7"] or c.endswith("_id") or "id" in c or "player" in c or "attacker" in c or "steam" in c:
            print(f"     num {c:20s} non_null={cnt['non_null']} steamid64_range_rows={cnt['steam_range']} ge_1e7_rows={cnt['ge_1e7']} distinct_max_per_file={cnt.get('distinct_max',0)}")
errs = [(d["path"].split("csds-cs2/")[-1][:90], d["error"][:110]) for d in rows if "error" in d]
print("errored files:", len(errs))
for e in errs:
    print("   ", e)
