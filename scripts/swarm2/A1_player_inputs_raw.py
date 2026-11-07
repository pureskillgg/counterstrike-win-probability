# A1: raw E: player_inputs row count per match, footer metadata only (pyarrow ParquetFile.metadata).
import os, json, time
import pyarrow.parquet as pq
import polars as pl
T = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
E = r"E:\csds_final\csds\2026\10"
t0 = time.time()
h = pl.read_parquet(os.path.join(T, "header", "dataframe_00000"), columns=["key"])
keys = h.get_column("key").to_list()
counts = {}
missing = 0
nonzero = 0
total = 0
for i, k in enumerate(keys):
    dd, mid = k.split("/")[3], k.split("/")[4]
    p = os.path.join(E, dd, mid, "player_inputs")
    if not os.path.exists(p):
        missing += 1
        counts[k] = None
        continue
    n = pq.ParquetFile(p).metadata.num_rows
    counts[k] = n
    total += n
    if n > 0:
        nonzero += 1
    if i % 400 == 0:
        print("progress", i, round(time.time() - t0, 1), flush=True)
vals = [v for v in counts.values() if v is not None]
print("matches", len(keys), "raw player_inputs file missing:", missing, "files with rows>0:", nonzero, "total rows:", total)
print("min/median/max rows per match:", min(vals), sorted(vals)[len(vals)//2], max(vals))
print("elapsed s", round(time.time() - t0, 1))
json.dump(counts, open(r"G:\pureskill\csds-cs2\logs\swarm2\A1_player_inputs_raw_counts.json", "w"))
