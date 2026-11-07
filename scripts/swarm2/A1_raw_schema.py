# A1: compare raw E: parquet schema (extensionless files) with tome header schema for picked matches.
# Key layout: csds/<YYYY>/<MM>/<DD>/<match_id>/csds  -> parts[3]=DD, parts[4]=match_id
import os, random
import polars as pl
import pyarrow.parquet as pq
T = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
E = r"E:\csds_final\csds\2026\10"
h = pl.read_parquet(os.path.join(T, "header", "dataframe_00000"), columns=["key", "match_id"])
keys = sorted(h.get_column("key").to_list())
# key -> match_id self-consistency (corrected index)
bad = [k for k, m in zip(h.get_column("key").to_list(), h.get_column("match_id").to_list()) if k.split("/")[4] != m]
print("key parts[4] == match_id mismatches:", len(bad), "of", h.height)
rng = random.Random(20261013)
picked = rng.sample(keys, 3)
print("picked:", picked)
for k in picked:
    dd, mid = k.split("/")[3], k.split("/")[4]
    base = os.path.join(E, dd, mid)
    p = os.path.join(base, "round_end")
    print(k, "| raw dir exists:", os.path.isdir(base), "| round_end exists:", os.path.exists(p))
mid = picked[0].split("/")[4]; dd = picked[0].split("/")[3]
base = os.path.join(E, dd, mid)
hs = pq.read_schema(os.path.join(base, "header")).names
ts = pl.read_parquet_schema(os.path.join(T, "header", "dataframe_00000"))
print("raw header cols:", len(hs), "tome header cols:", len(ts))
print("header cols only in raw:", sorted(set(hs) - set(ts.keys())))
print("header cols only in tome:", sorted(set(ts.keys()) - set(hs)))
print("round_end raw schema:", [(n, str(t)) for n, t in zip(pq.read_schema(os.path.join(base, "round_end")).names, pq.read_schema(os.path.join(base, "round_end")).types)])
