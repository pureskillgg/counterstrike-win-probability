"""Read-only peek at one csds manifest per year. Prints structure only."""
import gzip
import json
import sys
from collections import Counter

PATHS = [
    r"E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02\02\m_77fca7aed7217a25e7d1fea9\csds",
    r"E:\csds_final\csds\2026\10\01\m_a661aadb0b4b87e35cdec94c\csds",
]


def load(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    try:
        data = gzip.decompress(raw)
    except OSError:
        data = raw
    return json.loads(data.decode("utf-8"))


for p in PATHS:
    doc = load(p)
    print("==", p)
    print("top-level keys:", sorted(doc.keys()))
    chans = doc["channels"]
    print("channel entries:", len(chans))
    print("entry keys:", Counter(tuple(sorted(c.keys())) for c in chans).most_common(5))
    print("available values:", Counter(repr(c.get("available", "<absent>")) for c in chans))
    first = chans[0]
    print("first channel:", first["channel"])
    print("first columns (up to 3):", json.dumps(first["columns"][:3], indent=1)[:800])
    print()
