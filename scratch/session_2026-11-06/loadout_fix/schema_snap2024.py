import polars as pl, pyarrow.parquet as pq
P = "G:/pureskill/csds-cs2/data/combined/cache_snapshot_2024.parquet"
s = pq.read_schema(P)
names = s.names
print("n_cols", len(names), "n_rows", pq.read_metadata(P).num_rows)
print("cols", names)
