import glob, sys, pyarrow.parquet as pq
root = sys.argv[1]
for name in ["basic_player_death", "basic_round_end"]:
    files = sorted(glob.glob(root + "/" + name + ".*/dataframe_*"))
    schema = pq.read_schema(files[0])
    print("##", name, "pages:", len(files), "rows:", sum(pq.ParquetFile(f).metadata.num_rows for f in files))
    print(", ".join(f"{f.name}:{f.type}" for f in schema))
