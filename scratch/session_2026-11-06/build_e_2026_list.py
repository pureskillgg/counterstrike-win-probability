import csv
import os
import re
from collections import Counter

# Builds a list of player_personal paths on E: for train/val matches only.
# Excludes holdout/test labels and any key dated 2026-10-26 .. 2026-10-31.
MANIFEST = r"C:\Users\<user>\AppData\Local\Temp\claude\44708acea14da7d4\047ee5fa-b35b-4b25-97b1-beaf96834ff2\scratchpad\check_unchecked_oct\space\manifest_v2_labels.csv"
OUT = r"C:\Users\<user>\AppData\Local\Temp\claude\44708acea14da7d4\047ee5fa-b35b-4b25-97b1-beaf96834ff2\scratchpad\pii_paths_trainval_final2026.txt"
ROOT = "E:/csds_final"

DATE_RE = re.compile(r"^csds/(\d{4})/(\d{2})/(\d{2})/")

counts = Counter()
out_paths = []
with open(MANIFEST, encoding="utf-8") as fh:
    reader = csv.reader(fh)
    next(reader, None)
    for row in reader:
        if len(row) < 2:
            continue
        key, label = row[0], row[1]
        m = DATE_RE.match(key)
        if not m:
            counts["key_unparsed"] += 1
            continue
        year, month, day = m.group(1), m.group(2), m.group(3)
        if year != "2026":
            counts["not_2026"] += 1
            continue
        if label not in ("train", "val"):
            counts["label_" + label] += 1
            continue
        if month == "10" and day in {"26", "27", "28", "29", "30", "31"}:
            counts["excluded_oct26_31"] += 1
            continue
        counts["trainval_2026_candidates"] += 1
        base = key.rsplit("/", 1)[0]  # csds/2026/MM/DD/<uuid>
        p = ROOT + "/" + base + "/player_personal"
        if os.path.isfile(p):
            counts["present_on_E"] += 1
            out_paths.append(p)
        else:
            counts["absent_on_E"] += 1

with open(OUT, "w", encoding="utf-8") as fh:
    for p in out_paths:
        fh.write(p + "\n")

for k in sorted(counts):
    print(k + ":", counts[k])
print("written:", len(out_paths), "to", OUT)
