import glob, os, sys, pyarrow.parquet as pq
tome = os.path.join(sys.argv[1], "tome", "csds")
names = sorted(d for d in os.listdir(tome) if not d.startswith(".") and d != "header")
missing = []
for n in names:
    pages = sorted(glob.glob(os.path.join(tome, n, "dataframe_*")))
    for p in pages:
        s = pq.read_schema(p).names
        if "match_key" not in s:
            missing.append((n, os.path.basename(p), len(s)))
print("pages without match_key:", len(missing))
for m in missing:
    print("  ", m)
s = pq.read_schema(sorted(glob.glob(os.path.join(tome, "basic_player_spawn.*", "dataframe_*")))[0]).names
print("player_spawn columns:", s)
