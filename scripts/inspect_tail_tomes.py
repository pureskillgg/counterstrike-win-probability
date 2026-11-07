import glob, os, sys, pyarrow.parquet as pq
tome = os.path.join(sys.argv[1], "tome", "csds")
names = sorted(d for d in os.listdir(tome) if not d.startswith(".") and d != "header")
start = names.index([n for n in names if n.startswith("basic_player_spawn")][0]) if any(n.startswith("basic_player_spawn") for n in names) else 0
for n in names[start:]:
    pages = sorted(glob.glob(os.path.join(tome, n, "dataframe_*")))
    print("##", n, "pages:", len(pages), "rows:", sum(pq.ParquetFile(p).metadata.num_rows for p in pages))
    for p in pages[:1]:
        print("   cols:", pq.read_schema(p).names[:12], "...", len(pq.read_schema(p).names))
