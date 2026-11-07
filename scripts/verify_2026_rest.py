import glob, os, sys, pyarrow.parquet as pq
root = sys.argv[1]
tome = os.path.join(root, "tome", "csds")
hdr_keys = set()
for f in sorted(glob.glob(os.path.join(tome, "header", "dataframe_*"))):
    hdr_keys.update(pq.read_table(f, columns=["key"]).column("key").to_pylist())
n = "basic_player_inputs.2026-09-18,2026-10-08"
pages = sorted(glob.glob(os.path.join(tome, n, "dataframe_*")))
ks = sorted(glob.glob(os.path.join(tome, n, "keyset_*")))
print("player_inputs rows:", sum(pq.ParquetFile(p).metadata.num_rows for p in pages),
      "| keyset rows:", sum(pq.ParquetFile(p).metadata.num_rows for p in ks),
      "| page columns:", len(pq.read_schema(pages[0]).names))
print("keyset matches subset of header:", set(pq.read_table(ks[0]).column(0).to_pylist()) <= hdr_keys)
bad = []
for d in sorted(os.listdir(tome)):
    if d.startswith(".") or d == "header" or d == n:
        continue
    keys = set()
    for p in sorted(glob.glob(os.path.join(tome, d, "dataframe_*"))):
        keys.update(pq.read_table(p, columns=["match_key"]).column("match_key").to_pylist())
    if not keys <= hdr_keys:
        bad.append(d)
print("tomes with keys outside header (excluding player_inputs):", bad or "none")
