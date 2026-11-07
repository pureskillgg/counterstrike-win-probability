"""Learn the match_key format from a 2024-only file (no Oct 26-31 2026 rows possible).
Prints only generalised shapes and counts, never raw key values."""
import re
import sys
import collections
import pyarrow.parquet as pq
import pyarrow.compute as pc

path = sys.argv[1]
t = pq.read_table(path, columns=["match_key"])
col = t.column("match_key")
uniq = pc.unique(col).to_pylist()
print("distinct match_key:", len(uniq))
shapes = collections.Counter()
lens = collections.Counter()
for k in uniq:
    if k is None:
        shapes["<null>"] += 1
        continue
    s = re.sub(r"[0-9]", "9", k)
    s = re.sub(r"[a-z]", "a", s)
    s = re.sub(r"[A-Z]", "A", s)
    shapes[s] += 1
    lens[len(k)] += 1
print("distinct shapes:", len(shapes))
for s, c in shapes.most_common(5):
    print("  shape:", s[:60], "count:", c)
print("length histogram:", dict(lens.most_common(5)))
ex = uniq[0] if uniq else ""
print("masked example:", ex[:3] + "***")
