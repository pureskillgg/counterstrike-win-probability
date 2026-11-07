import sys, re, collections
import pyarrow.parquet as pq
path = sys.argv[1]
t = pq.read_table(path, columns=["match_key"])
keys = sorted(set(x for x in t.column("match_key").to_pylist() if x is not None))
segs = [k.split("/") for k in keys]
print("segment counts per key:", collections.Counter(len(s) for s in segs))
print("segment 0 distinct:", len(set(s[0] for s in segs)), "values:", sorted(set(s[0] for s in segs)))
print("segment 0 char classes lowercase-only:", all(re.fullmatch(r"[a-z]+", s[0]) for s in segs))
print("segment 5 distinct:", len(set(s[5] for s in segs)), "values:", sorted(set(s[5] for s in segs)))
print("segment 1 (year) distinct:", sorted(collections.Counter(s[1] for s in segs).items()))
print("segment 2 (month) distinct:", sorted(set(s[2] for s in segs)))
print("segment 3 (day) distinct count:", len(set(s[3] for s in segs)))
mid = [s[4] for s in segs]
print("middle seg distinct:", len(set(mid)), "lengths:", dict(collections.Counter(len(m) for m in mid)))
print("middle seg charset: only [0-9a-z-] ->", all(re.fullmatch(r"[0-9a-z\-]+", m) for m in mid))
print("middle seg has uppercase ->", any(re.search(r"[A-Z]", m) for m in mid))
