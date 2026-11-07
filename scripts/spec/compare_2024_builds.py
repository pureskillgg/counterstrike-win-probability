import os, pyarrow.parquet as pq, glob
base = r"G:\pureskill\csds-cs2"
a = os.path.join(base, "tomes", "2024-02-all", "tome", "csds")
b = os.path.join(base, "tomes", "2024-02-g", "tome", "csds")
def rows(root, name):
    fs = sorted(glob.glob(os.path.join(root, name, "dataframe_*")))
    return sum(pq.ParquetFile(f).metadata.num_rows for f in fs)
names_a = sorted(n for n in os.listdir(a) if n.startswith("basic_") or n == "header")
names_b = set(n for n in os.listdir(b) if n.startswith("basic_") or n == "header")
bad = 0
for n in names_a:
    ra = rows(a, n)
    rb = rows(b, n) if n in names_b else None
    ok = rb is not None and ra == rb
    bad += (not ok)
    print(f"{n:45s} E={ra:>10d} G={str(rb):>10s} {'OK' if ok else 'DIFF'}")
print("tomes E:", len(names_a), " G:", len(names_b), " mismatched:", bad)
