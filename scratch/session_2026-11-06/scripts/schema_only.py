import sys, os, json
import pyarrow.parquet as pq
base = sys.argv[1]
for name in ["cache_snapshot_2026.parquet","deaths.parquet","grid.parquet","matches.parquet","rounds.parquet"]:
    p = os.path.join(base, name)
    f = pq.ParquetFile(p)
    md = f.metadata
    print(f"== {name}: rows={md.num_rows} cols={md.num_columns} row_groups={md.num_row_groups}")
    sch = f.schema_arrow
    for fld in sch:
        print(f"   {fld.name}: {fld.type}")
