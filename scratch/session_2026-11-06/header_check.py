import sys
from collections import Counter

import pyarrow.parquet as pq

# Aggregate-only check of header free-text fields. Prints counts and shape
# classes only (letters -> a/A, digits -> 9). No raw value is printed.
paths_file = sys.argv[1]
with open(paths_file, encoding="utf-8") as fh:
    paths = [line.strip() for line in fh if line.strip()]

FIELDS = ["client_name", "server_name", "sharecode", "demo_id", "providence", "platform", "match_date", "addons"]


def shape(s):
    res = []
    for ch in s:
        if ch.isdigit():
            c = "9"
        elif ch.isalpha() and ch.islower():
            c = "a"
        elif ch.isalpha():
            c = "A"
        else:
            c = ch
        if not res or res[-1] != c or c not in ("9", "a", "A"):
            res.append(c)
    return "".join(res)


distinct = {f: Counter() for f in FIELDS}
shapes = {f: Counter() for f in FIELDS}
space_count = Counter()
files = 0
errors = 0
for path in paths:
    base = path.rsplit("/", 1)[0]
    try:
        tbl = pq.read_table(base + "/header")
    except Exception:
        errors += 1
        continue
    files += 1
    for f in FIELDS:
        if f not in tbl.column_names:
            continue
        for v in tbl.column(f).to_pylist():
            if v is None:
                distinct[f]["<null>"] += 1
                continue
            distinct[f][v] += 1
            shapes[f][shape(v)[:40]] += 1
            if " " in v.strip():
                space_count[f] += 1

print("header files read:", files, "errors:", errors)
for f in FIELDS:
    d = distinct[f]
    print(f"{f}: distinct_values={len(d)} total_rows={sum(d.values())}")
    print("   top shape classes (shape: count):", [(k, v) for k, v in shapes[f].most_common(4)])
    print("   rows containing a space:", space_count[f])
    print("   rows per distinct value (top 5):", sorted(d.values(), reverse=True)[:5])
