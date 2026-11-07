import json, sys
import polars as pl
import pyarrow.parquet as pq
keys = json.load(open(r"C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/loadout_fix/match_keys.json"))
print("match_keys type", type(keys).__name__, "len", len(keys), "unique", len(set(keys)))
print("first", keys[0]); print("last", keys[-1])
base = "G:/pureskill/csds-cs2/data/"
for d, f in [("status_1hz","2026-09-01.parquet"),("status_1hz_2024","2024-02-01.parquet"),("status_16hz","2026-09-01.parquet"),("status_8hz","2026-09-01.parquet"),("status_random","2026-09-01.parquet"),("status_8new","2026-09-01.parquet")]:
    p = base + d + "/" + f
    md = pq.ParquetFile(p).metadata
    print("==", d, f, "rows", md.num_rows, "rowgroups", md.num_row_groups, "cols", md.num_columns)
    print("  schema:", [(md.schema.column(i).name, md.schema.column(i).physical_type) for i in range(md.num_columns)])
