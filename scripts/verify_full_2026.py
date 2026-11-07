import glob, os, sys, pyarrow.parquet as pq, pyarrow as pa
root = sys.argv[1]
tome = os.path.join(root, "tome", "csds")
names = sorted(d for d in os.listdir(tome) if not d.startswith("."))
print("tomes:", len(names))
hdr_keys = set()
for f in sorted(glob.glob(os.path.join(tome, "header", "dataframe_*"))):
    hdr_keys.update(pq.read_table(f, columns=["key"]).column("key").to_pylist())
print("header keys:", len(hdr_keys))
bad = []
for n in names:
    if n == "header":
        continue
    pages = sorted(glob.glob(os.path.join(tome, n, "dataframe_*")))
    ks = sorted(glob.glob(os.path.join(tome, n, "keyset_*")))
    rows = sum(pq.ParquetFile(p).metadata.num_rows for p in pages)
    kset = sum(pq.ParquetFile(p).metadata.num_rows for p in ks)
    keys = set()
    for p in pages:
        keys.update(pq.read_table(p, columns=["match_key"]).column("match_key").to_pylist())
    ok = keys <= hdr_keys
    if not ok:
        bad.append(n)
    print(f"{n}: pages={len(pages)} rows={rows} keyset_rows={kset} matches_with_rows={len(keys)} subset_of_header={ok}")
print("problems:", bad if bad else "none")
