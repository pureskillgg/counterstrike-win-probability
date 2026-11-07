import os
import re
import collections
import json

SCR = "C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad"
SOURCES = {
    "cs2data": os.path.join(SCR, "raw_cs2data.txt"),
    "cs2data_gz": os.path.join(SCR, "raw_cs2data_gz.txt"),
}
OUT = os.path.join(SCR, "volume_cs2data_objects.txt")
LINE = re.compile(r"^(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2})\s+(\d+) (.+)$")


def human(n):
    v = float(n)
    for u in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if v < 1024 or u == "TiB":
            return f"{int(n):,} B" if u == "B" else f"{v:,.2f} {u}"
        v /= 1024


def ext_of(name):
    parts = name.split(".")
    if len(parts) == 1:
        return "(none)"
    if parts[-1] == "gz" and len(parts) >= 3:
        return "." + parts[-2] + ".gz"
    return "." + parts[-1]


rows = {}
for prefix, path in SOURCES.items():
    items = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\r\n")
            if not line.strip():
                continue
            m = LINE.match(line)
            if not m:
                raise SystemExit(f"unparsed line in {prefix}: {line[:120]}")
            items.append({
                "key": m.group(4),
                "bytes": int(m.group(3)),
                "mtime": m.group(1) + " " + m.group(2),
            })
    rows[prefix] = items

# Write the combined list: one line per object, key<TAB>bytes, both prefixes.
with open(OUT, "w", encoding="utf-8", newline="\n") as f:
    for prefix in SOURCES:
        for it in rows[prefix]:
            f.write(f"{it['key']}\t{it['bytes']}\n")

summary = {}
for prefix, items in rows.items():
    total_bytes = sum(i["bytes"] for i in items)
    top = collections.defaultdict(lambda: [0, 0])
    second = collections.defaultdict(lambda: [0, 0])
    exts = collections.defaultdict(lambda: [0, 0])
    for it in items:
        rel = it["key"][len(prefix) + 1:]
        parts = rel.split("/")
        if len(parts) == 1:
            t = "(root file) " + parts[0]
        else:
            t = parts[0]
        top[t][0] += 1
        top[t][1] += it["bytes"]
        if parts[0] == "data" and len(parts) > 2:
            s = "data/" + parts[1]
            second[s][0] += 1
            second[s][1] += it["bytes"]
        e = ext_of(parts[-1])
        exts[e][0] += 1
        exts[e][1] += it["bytes"]
    summary[prefix] = {
        "objects": len(items),
        "total_bytes": total_bytes,
        "total_human": human(total_bytes),
        "top": {k: {"count": v[0], "bytes": v[1], "human": human(v[1])} for k, v in sorted(top.items())},
        "second_under_data": {k: {"count": v[0], "bytes": v[1], "human": human(v[1])} for k, v in sorted(second.items())},
        "extensions": {k: {"count": v[0], "bytes": v[1], "human": human(v[1])} for k, v in sorted(exts.items(), key=lambda kv: -kv[1][1])},
        "first5_keys": [i["key"] for i in items[:5]],
    }

# Cross-prefix comparison on a logical name (relative path, with data/ and .gz removed).
def logical(prefix, key):
    rel = key[len(prefix) + 1:]
    if rel.startswith("data/"):
        rel = rel[len("data/"):]
    if rel.endswith(".gz"):
        rel = rel[:-3]
    return rel

a = {logical("cs2data", i["key"]): i for i in rows["cs2data"]}
b = {logical("cs2data_gz", i["key"]): i for i in rows["cs2data_gz"]}
both = sorted(set(a) & set(b))
only_a = sorted(set(a) - set(b))
only_b = sorted(set(b) - set(a))
overlap = []
for name in both:
    overlap.append({
        "name": name,
        "cs2data_bytes": a[name]["bytes"],
        "cs2data_mtime": a[name]["mtime"],
        "gz_key": b[name]["key"],
        "gz_bytes": b[name]["bytes"],
        "gz_mtime": b[name]["mtime"],
    })

out = {
    "summary": summary,
    "overlap_same_logical_name": overlap,
    "only_in_cs2data_count": len(only_a),
    "only_in_cs2data_bytes": sum(a[n]["bytes"] for n in only_a),
    "only_in_gz_count": len(only_b),
    "only_in_gz_bytes": sum(b[n]["bytes"] for n in only_b),
    "only_in_cs2data_names": only_a,
    "only_in_gz_names": only_b,
    "list_file": OUT,
    "list_lines": sum(len(v) for v in rows.values()),
}
print(json.dumps(out, indent=1))
